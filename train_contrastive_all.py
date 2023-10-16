import argparse
import datetime
import time
import os
from tqdm import tqdm
import json

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import aug_utils as transforms
from utils import save_checkpoint, adjust_learning_rate, get_root_logger, set_seed
from tree_dataset import (
    NeuronTreeDataset,
    NeuronTreeDatasetTwoViews,
    get_collate_fn,
    LABEL_DICT,
)
from moco import MoCoTreeLSTM


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def test(net, memory_data_loader, test_data_loader, epoch, args, dataset_name):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for b in memory_data_loader:
            b0 = b
            break
        net(b0)  # dummy forward
        for batch in memory_data_loader:
            feature = net(batch)
            feature = torch.nn.functional.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(
            memory_data_loader.dataset.targets, device=feature_bank.device
        )
        # loop test data to predict the label by weighted knn search
        # test_bar = test_data_loader
        net(b0)  # dummy forward
        for batch in test_data_loader:
            feature = net(batch)
            feature = torch.nn.functional.normalize(feature, dim=1)
            k = 5 if memory_data_loader.dataset == "JM" else args.knn_k
            pred_labels = knn_predict(
                feature, feature_bank, feature_labels, classes, k, args.knn_t
            )
            total_num += feature.size(0)
            total_top1 += (
                (pred_labels[:, 0] == batch.label.to(torch.device("cuda")))
                .float()
                .sum()
                .item()
            )
    net.train()
    return total_top1 / total_num * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Contrastive Training")
    # Basic
    parser.add_argument(
        "--work_dir", type=str, default="./work_dir", help="experiment root"
    )
    parser.add_argument(
        "--exp_name", type=str, default="debug", required=True, help="experiment name"
    )
    parser.add_argument("--dataset", type=str, default="all_wo_others")
    parser.add_argument("--data_dir", type=str, default="data/raw/bil")
    parser.add_argument("--label_dict", type=str, default="all_wo_others")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Model
    parser.add_argument("--model", type=str, default="double")
    parser.add_argument("--child_mode", type=str, default="sum", help="[sum, average]")
    parser.add_argument(
        "--input_features",
        nargs="+",
        type=int,
        default=[2, 3, 4, 12, 13],
        help="selected columns",
    )
    parser.add_argument("--h_size", type=int, default=128, help="memory size for lstm")
    parser.add_argument("--bn", action="store_true", default=False)
    parser.add_argument("--projector_bn", action="store_true", default=False)
    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="batch size in training [default: 128]",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="number of epoch in training [default: 200]",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.06,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--start_epoch", type=int, default=0, help="starting_epoch")
    parser.add_argument(
        "--save_freq", type=int, default=5, help="Saving frequency [default: 5]"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument(
        "--schedule",
        default=[120, 160],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on",
    )
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")
    parser.add_argument(
        "--wd", default=5e-4, type=float, metavar="W", help="weight decay"
    )

    # MoCo specific configs:
    parser.add_argument("--moco-dim", default=128, type=int, help="feature dimension")
    parser.add_argument(
        "--moco-k", default=1024, type=int, help="queue size; number of negative keys"
    )
    parser.add_argument(
        "--moco-m",
        default=0.99,
        type=float,
        help="moco momentum of updating key encoder",
    )
    parser.add_argument("--moco-t", default=0.1, type=float, help="softmax temperature")
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=False,
        help="use a symmetric loss function that backprops to both views",
    )

    # Evaluation
    parser.add_argument(
        "--val_freq", type=int, default=5, help="Val frequency [default: 5]"
    )
    # Augmentation
    parser.add_argument("--aug_scale_feats", action="store_true", default=False)
    parser.add_argument("--aug_scale_coords", action="store_true", default=False)
    parser.add_argument("--aug_rotate", action="store_true", default=False)
    parser.add_argument("--aug_jitter_coords", action="store_true", default=False)
    parser.add_argument("--aug_shift_coords", action="store_true", default=False)
    parser.add_argument("--aug_flip", action="store_true", default=False)
    parser.add_argument("--aug_mask_feats", action="store_true", default=False)
    parser.add_argument("--aug_jitter_length", action="store_true", default=False)
    parser.add_argument("--aug_elasticate", action="store_true", default=False)
    parser.add_argument("--aug_drop_tree", action="store_true", default=False)
    parser.add_argument("--aug_skip_parent_node", action="store_true", default=False)
    parser.add_argument(
        "--aug_swap_sibling_subtrees", action="store_true", default=False
    )
    parser.add_argument(
        "--drop_tree_prob",
        nargs="+",
        type=float,
        default=[0, 0, 0.005],
        help="drop_tree probability",
    )
    parser.add_argument(
        "--use_translation_feats",
        action="store_true",
        default=False,
        help="use 24-d translation feats",
    )
    # knn monitor
    parser.add_argument("--knn", action="store_true", default=False)
    parser.add_argument("--knn-k", default=20, type=int, help="k in kNN monitor")
    parser.add_argument(
        "--knn-t",
        default=0.5,
        type=float,
        help="softmax temperature in kNN monitor; could be different with moco-t",
    )
    parser.add_argument("--debug", action="store_true")

    # clustering monitor
    parser.add_argument("--kmeans", action="store_true", default=False)
    parser.add_argument(
        "--eval_part", type=str, default="full", help="[full | backbone]"
    )

    # training dataset
    parser.add_argument(
        "--train_split", type=str, default="full", help="[train | full]"
    )
    parser.add_argument(
        "--use_balanced_memory_data",
        action="store_true",
        default=False,
        help="using balanced memory data for knn evaluation",
    )
    # eval dataset
    parser.add_argument("--eval_jm", action="store_true", default=False)
    parser.add_argument("--eval_act", action="store_true", default=False)
    # parser.add_argument('--eval_seu', action='store_true', default=False)

    args = parser.parse_args()
    set_seed(args.seed)
    assert args.aug_scale_feats is False, "current disable aug_scale_feats"
    # hacking the args
    if args.use_translation_feats:
        args.input_features += [i for i in range(20, 44)]
    args.work_dir = f"{args.work_dir}/{args.exp_name}"
    # create work_dir
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.debug:
        log_file = None
        args.save_freq = 10000
        args.val_freq = 1
    else:
        log_file = f"{args.work_dir}/train_{timestamp}.log"
    logger = get_root_logger(log_file=log_file, log_level="INFO")
    # hyper parameters
    device = torch.device("cuda")
    loader_device = torch.device("cuda")
    h_size = args.h_size
    epochs = args.epochs

    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))  # print args
    logger.info("=> creating models ...")

    # create the model
    model = MoCoTreeLSTM(
        args,  # for TreeLSTM
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        symmetric=args.symmetric,
    ).to(device)
    logger.info(model)
    # create the optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9
    )
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    aug_switchs = [
        args.aug_scale_feats,
        args.aug_scale_coords,
        args.aug_rotate,
        args.aug_jitter_coords,
        args.aug_shift_coords,
        args.aug_flip,
        args.aug_mask_feats,
        args.aug_jitter_length,
        args.aug_elasticate,
    ]
    aug_fns = [
        transforms.RandomScaleFeats(p=0.2),
        transforms.RandomScaleCoords(p=0.2)
        if not args.use_translation_feats
        else transforms.RandomScaleCoordsTranslation(p=0.2),
        transforms.RandomRotate(p=0.5),
        transforms.RandomJitter(p=0.2),
        transforms.RandomShift(p=0.2),
        transforms.RandomFlip(p=1),
        transforms.RandomMaskFeats(p=0.2),
        transforms.RandomJitterLength(p=0.2),
        transforms.RandomElasticate(p=0.2),
    ]
    feat_augs = [aug_fns[i] for i in range(len(aug_switchs)) if aug_switchs[i] is True]
    if len(feat_augs) == 0:
        feat_augs = None
    else:
        feat_augs = transforms.Compose(feat_augs)
    topo_aug_swtichs = [
        args.aug_drop_tree,
        args.aug_skip_parent_node,
        args.aug_swap_sibling_subtrees,
    ]
    topo_aug_fns = [
        transforms.RandomDropSubTrees(probs=[0.05], max_cnt=5),
        transforms.RandomSkipParentNode(probs=[0.05], max_cnt=10),
        transforms.RandomSwapSiblingSubTrees(probs=[0.05], max_cnt=10),
    ]
    topo_augs = [
        topo_aug_fns[i]
        for i in range(len(topo_aug_swtichs))
        if topo_aug_swtichs[i] is True
    ]
    if len(topo_augs) == 0:
        topology_transformations = None
    else:
        topology_transformations = transforms.Compose(topo_augs)
    logger.info("=====>using augmentations")
    logger.info(feat_augs)
    logger.info(topology_transformations)

    trainset = NeuronTreeDatasetTwoViews(
        phase="full" if args.dataset == "all_wo_others" else "train",
        dataset=args.dataset,
        label_dict=LABEL_DICT["all_wo_others"]
        if args.dataset == "rebuttal"
        else LABEL_DICT[args.dataset],
        data_dir=args.data_dir,
        topology_transformations=topology_transformations,
        attribute_transformations=feat_augs,
        input_features=args.input_features,
    )

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        collate_fn=get_collate_fn(loader_device, use_two_views=True),
        shuffle=True,
        drop_last=True,
    )

    bil_memory_data = NeuronTreeDataset(
        phase="train",
        dataset="bil_6_classes"
        if not args.use_balanced_memory_data
        else "bil_6_classes_balanced",
        data_dir=args.data_dir,
        label_dict=LABEL_DICT["bil_6_classes"],
        input_features=args.input_features,
    )

    bil_memory_loader = DataLoader(
        dataset=bil_memory_data,
        batch_size=args.batch_size,
        collate_fn=get_collate_fn(loader_device),
        shuffle=False,
    )
    bil_testset = NeuronTreeDataset(
        phase="test",
        dataset="bil_6_classes",
        label_dict=LABEL_DICT["bil_6_classes"],
        input_features=args.input_features,
    )
    bil_test_loader = DataLoader(
        dataset=bil_testset,
        batch_size=args.batch_size,
        collate_fn=get_collate_fn(loader_device),
        shuffle=False,
    )
    memory_loaders = [bil_memory_loader]
    test_loaders = [bil_test_loader]
    test_datasets = ["BIL"]
    if args.eval_jm:
        JM_memory_data = NeuronTreeDataset(
            phase="train",
            dataset="JM" if not args.use_balanced_memory_data else "JM_balanced",
            label_dict=LABEL_DICT["JM"],
            data_dir=args.data_dir,
            input_features=args.input_features,
        )
        JM_memory_loader = DataLoader(
            dataset=JM_memory_data,
            batch_size=args.batch_size,
            collate_fn=get_collate_fn(loader_device),
            shuffle=False,
        )
        JM_testset = NeuronTreeDataset(
            phase="test",
            dataset="JM",
            label_dict=LABEL_DICT["JM"],
            input_features=args.input_features,
        )
        JM_test_loader = DataLoader(
            dataset=JM_testset,
            batch_size=args.batch_size,
            collate_fn=get_collate_fn(loader_device),
            shuffle=False,
        )
        memory_loaders.append(JM_memory_loader)
        test_loaders.append(JM_test_loader)
        test_datasets.append("JM")

    if args.eval_act:
        ACT_memory_data = NeuronTreeDataset(
            phase="train",
            dataset="ACT" if not args.use_balanced_memory_data else "ACT_balanced",
            label_dict=LABEL_DICT["ACT"],
            input_features=args.input_features,
        )
        ACT_memory_loader = DataLoader(
            dataset=ACT_memory_data,
            batch_size=args.batch_size,
            collate_fn=get_collate_fn(loader_device),
            shuffle=False,
        )
        ACT_testset = NeuronTreeDataset(
            phase="test",
            dataset="ACT",
            label_dict=LABEL_DICT["ACT"],
            input_features=args.input_features,
        )
        ACT_test_loader = DataLoader(
            dataset=ACT_testset,
            batch_size=args.batch_size,
            collate_fn=get_collate_fn(loader_device),
            shuffle=False,
        )
        memory_loaders.append(ACT_memory_loader)
        test_loaders.append(ACT_test_loader)
        test_datasets.append("ACT")

    total_iters = len(train_loader) * (epochs)
    current_iter = 0
    # training loop
    best_test_acc, best_acc_epoch = [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]
    start_time = time.time()
    for epoch in range(args.start_epoch + 1, epochs + 1):
        model.train()
        adjust_learning_rate(optimizer, epoch, args)
        for step, batch in enumerate(train_loader):
            try:
                loss = model(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except:
                logger.info("bad thing happens")
                pass
            current_iter += 1
            current_time = time.time()
            elapsed = current_time - start_time
            logger.info(
                f"Train Epoch {epoch:03d} | Step {step:03d} | Loss {loss.item():.4f} | "
                f"Elasped {str(datetime.timedelta(seconds=elapsed))[:-7]} | "
                f"ETA {str(datetime.timedelta(seconds=elapsed/current_iter * (total_iters-current_iter)))[:-7]}"
            )
        if epoch % args.val_freq == 0:
            if args.knn:
                parts = ["backbone", "full"]
                for j, eval_part in enumerate([model.backbone_q, model.encoder_q]):
                    for ix, (memory_loader, test_loader) in enumerate(
                        zip(memory_loaders, test_loaders)
                    ):
                        test_acc = test(
                            eval_part,
                            memory_loader,
                            test_loader,
                            epoch,
                            args,
                            test_datasets[ix],
                        )
                        if test_acc > best_test_acc[j][ix]:
                            best_test_acc[j][ix], best_acc_epoch[j][ix] = (
                                test_acc,
                                epoch,
                            )
                        logger.info(
                            f" Test Epoch {epoch:03d} | {test_datasets[ix]}-{parts[j]} | KNN Acc: {test_acc:.2f}% | Best Acc {best_test_acc[j][ix]:.2f}% at epoch {best_acc_epoch[j][ix]} "
                        )
        if epoch % args.save_freq == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=f"{args.work_dir}/epoch_{epoch}.pth",
            )
            logger.info(
                f"saving checkpoint at epoch {epoch} to {args.work_dir}/epoch_{epoch}.pth"
            )
