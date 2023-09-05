import time, datetime
import argparse
import os
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import aug_utils as transforms
from utils import save_checkpoint, get_root_logger, set_seed, adjust_learning_rate
from tree_dataset import NeuronTreeDataset, get_collate_fn, LABEL_DICT
from treelstm import TreeLSTMv2, TreeLSTM, TreeLSTMDouble, TreeLSTM_wo_MLP


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Supervised Training")
    # Basic
    parser.add_argument(
        "--work_dir", type=str, default="./work_dir", help="experiment root"
    )
    parser.add_argument(
        "--exp_name", type=str, default="debug", required=True, help="experiment name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="seu_6_classes",
        help="[seu_6_classes | JM | ACT]",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # Model
    parser.add_argument("--model", type=str, default="ori")
    parser.add_argument("--bn", action="store_true", default=False)
    parser.add_argument("--child_mode", type=str, default="sum", help="[sum, mean]")
    parser.add_argument(
        "--input_features",
        nargs="+",
        type=int,
        default=[2, 3, 4, 12, 13],
        help="selected columns",
    )
    parser.add_argument("--h_size", type=int, default=128, help="memory size for lstm")
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
        "--lr", default=1e-4, type=float, help="init learning rate [default: 1e-4]"
    )
    parser.add_argument("--start_epoch", type=int, default=0, help="starting_epoch")
    parser.add_argument(
        "--save_freq", type=int, default=10, help="Saving frequency [default: 1]"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    # Evaluation
    parser.add_argument(
        "--val_freq", type=int, default=5, help="Val frequency [default: 5]"
    )
    parser.add_argument(
        "--schedule",
        default=[60],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
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
    parser.add_argument(
        "--drop_tree_prob",
        nargs="+",
        type=float,
        default=[0, 0, 0.005],
        help="drop_tree probability",
    )
    # others
    parser.add_argument(
        "--use_translation_feats",
        action="store_true",
        default=False,
        help="use 24-d translation feats",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--use_balanced_memory_data",
        action="store_true",
        default=False,
        help="using balanced memory data for knn evaluation",
    )
    # linear evaluation
    parser.add_argument(
        "--mode", type=str, default="finetune", help="random_init | frozon | finetune"
    )
    parser.add_argument(
        "--pretrained", default="", type=str, help="path to pretrained checkpoint"
    )

    args = parser.parse_args()
    args.cos = False
    set_seed(args.seed)

    if args.use_translation_feats:
        args.input_features += [i for i in range(20, 44)]
    if args.use_balanced_memory_data:
        _dataset = f"{args.dataset}_balanced"
    else:
        _dataset = args.dataset
    args.work_dir = f"{args.work_dir}/{args.exp_name}"
    # create work_dir
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.debug:
        log_file = None
        args.save_freq = 10000
    else:
        log_file = f"{args.work_dir}/train_{timestamp}.log"
    logger = get_root_logger(log_file=log_file, log_level="INFO")

    # hyper parameters
    device = torch.device("cuda")
    h_size = args.h_size
    epochs = args.epochs
    if args.model == "ori":
        model = TreeLSTM
    elif args.model == "one_cell":
        model = TreeLSTM_wo_MLP
    elif args.model == "ori_bn":
        model = TreeLSTMv2
        args.bn = True
    elif args.model == "double":
        model = TreeLSTMDouble
        args.bn = True

    # create the model
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))  # print args
    logger.info("=> creating models ...")

    model = model(
        len(args.input_features),
        h_size,
        len(LABEL_DICT[args.dataset]),
        mode=args.child_mode,
        bn=args.bn,
    ).to(device)
    logger.info(model)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.mode == "frozen":
        for name, param in model.named_parameters():
            if name not in ["linear.weight", "linear.bias"]:
                param.requires_grad = False
        # init the fc layer
        model.linear.weight.data.normal_(mean=0.0, std=0.01)
        model.linear.bias.data.zero_()

    if args.mode != "random_init":
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                logger.info("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint["state_dict"]
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith("backbone_q"):
                        # remove prefix
                        state_dict[k[len("backbone_q.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"linear.weight", "linear.bias"}

                logger.info("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.pretrained))

            if args.mode == "frozen":
                # optimize only the linear classifier
                parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
                assert len(parameters) == 2  # linear.weight, linear.bias
                # optimizer = torch.optim.SGD(parameters, args.lr,
                # momentum=0.9)
                optimizer = torch.optim.Adam(parameters, lr=args.lr)
                logger.info("=> using frozen backbone")
            elif args.mode == "finetune":
                args.lr = 2e-3
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=0.01
                )
                logger.info("=> fine-tuning the model")
            else:
                raise ValueError(f"get invalid args.mode {args.mode}")
        else:
            logger.info("=> if not random_init, a checkpoint must be given")
    else:
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0.01)
        logger.info("=> random init.")
        # optimizer = torch.optim.SGD(model.parameters(), args.lr,
        # momentum=0.9)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

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
    if args.aug_drop_tree:
        topology_transformations = transforms.Compose(
            [transforms.RandomDropSubTrees(args.drop_tree_prob)]
        )
    else:
        topology_transformations = None
    logger.info("=====>using augmentations")
    logger.info(feat_augs)
    logger.info(topology_transformations)

    trainset = NeuronTreeDataset(
        phase="train",
        dataset=_dataset,
        label_dict=LABEL_DICT[args.dataset],
        topology_transformations=topology_transformations,
        attribute_transformations=feat_augs,
        input_features=args.input_features,
    )
    testset = NeuronTreeDataset(
        phase="test",
        dataset=args.dataset,
        label_dict=LABEL_DICT[args.dataset],
        input_features=args.input_features,
    )
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        collate_fn=get_collate_fn(device, use_two_views=False),
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=testset,
        batch_size=args.batch_size * 2,
        collate_fn=get_collate_fn(device),
        shuffle=True,
    )

    total_iters = len(train_loader) * (epochs)
    current_iter = 0
    best_test_acc, best_epoch, is_best = 0, 0, False
    # training loop
    start_time = time.time()
    for module in model.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm1d):
            if hasattr(module, "weight"):
                module.weight.requires_grad_(False)
            if hasattr(module, "bias"):
                module.bias.requires_grad_(False)
            # module.track_running_stats=False
    for epoch in range(args.start_epoch + 1, epochs + 1):
        epoch_counts = 0
        epoch_correct = 0
        model.train()

        for b in test_loader:
            b0 = b
            break
        model(b0)  # dummy forward
        if args.mode == "finetune":
            adjust_learning_rate(optimizer, epoch, args)
        for step, batch in enumerate(train_loader):
            logits = model(batch)
            loss = criterion(logits, batch.label.cuda())
            pred = torch.argmax(logits, 1)
            acc = float(torch.sum(torch.eq(batch.label.cuda(), pred))) / len(
                batch.label
            )
            epoch_correct += len(batch.label) * acc
            epoch_counts += len(batch.label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_iter += 1
            current_time = time.time()
            elapsed = current_time - start_time
            logger.info(
                f"Train Epoch {epoch:03d} | Step {step:03d} | Loss {loss.item():.4f} | "
                f"Batch Acc {acc*100:.2f} | Epoch Acc {epoch_correct*100/epoch_counts:.2f} | "
                f"Elasped {str(datetime.timedelta(seconds=elapsed))[:-7]} | "
                f"ETA {str(datetime.timedelta(seconds=elapsed/current_iter * (total_iters-current_iter)))[:-7]}"
            )
        if epoch % args.val_freq == 0:
            model.eval()
            test_counts = 0
            test_correct = 0
            for b in test_loader:
                b0 = b
                break
            model(b0)  # dummy forward
            for step, batch in enumerate(test_loader):
                logits = model(batch)
                loss = criterion(logits, batch.label.cuda())
                pred = torch.argmax(logits, 1)
                acc = float(torch.sum(torch.eq(batch.label.cuda(), pred))) / len(
                    batch.label
                )
                test_correct += len(batch.label) * acc
                test_counts += len(batch.label)
            test_acc = test_correct * 100 / test_counts
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
            logger.info(
                f" Test Epoch {epoch:03d} | {args.dataset} | Acc {test_acc:.2f} | Best Acc {best_test_acc:.2f} at epoch {best_epoch}"
            )
            # could be a bug, saving model only after test.
            if epoch % args.save_freq == 0 or is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=is_best,
                    filename=f"{args.work_dir}/epoch_{epoch}.pth",
                )
                logger.info(
                    f"saving checkpoint at epoch {epoch} to {args.work_dir}/epoch_{epoch}.pth"
                )
