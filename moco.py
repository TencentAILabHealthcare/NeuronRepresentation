import torch
import torch.nn as nn
from treelstm import TreeLSTM, TreeLSTMv2, TreeLSTMDouble


class MoCoTreeLSTM(nn.Module):
    def __init__(self, args, dim=128, K=256, m=0.99, T=0.1, symmetric=True):
        super(MoCoTreeLSTM, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        if args.model == "ori":
            model = TreeLSTM
        elif args.model == "v2":
            model = TreeLSTMv2
        elif args.model == "double":
            model = TreeLSTMDouble
        # create the encoders
        self.backbone_q = model(
            x_size=len(args.input_features),
            h_size=args.h_size,
            num_classes=0,  # fc unused
            fc=False,
            bn=args.bn,
            mode=args.child_mode,
        )
        self.backbone_k = model(
            x_size=len(args.input_features),
            h_size=args.h_size,
            num_classes=0,  # fc unused
            fc=False,
            bn=args.bn,
            mode=args.child_mode,
        )

        dim_mlp = args.h_size
        if args.projector_bn:
            self.projector_q = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp, affine=False),
                nn.ReLU(),
                nn.Linear(dim_mlp, dim),
            )
            self.projector_k = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp, affine=False),
                nn.ReLU(),
                nn.Linear(dim_mlp, dim),
            )
        else:
            self.projector_q = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
            )
            self.projector_k = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
            )
        self.encoder_q = nn.Sequential(self.backbone_q, self.projector_q)
        self.encoder_k = nn.Sequential(self.backbone_k, self.projector_k)

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        if self.K % keys.shape[0] != 0:
            # should be sufficient to handle boundary cases.
            print("before", keys.shape)
            keys = torch.cat([keys, keys[-1:]], dim=0)
            print("after", keys.shape)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, view1, view2):
        # compute query features
        q = self.encoder_q(view1)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            k = self.encoder_k(view2)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, batch):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        view1, view2 = batch.view1, batch.view2
        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(view1, view2)
            loss_21, q2, k1 = self.contrastive_loss(view2, view1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(view1, view2)

        self._dequeue_and_enqueue(k)

        return loss
