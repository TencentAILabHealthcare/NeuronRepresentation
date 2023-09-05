from clustering import Kmeans

# from fast_pytorch_kmeans import KMeans

import argparse
import os
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tree_dataset import NeuronTreeDataset, SEU_6_classes, get_collate_fn, LABEL_DICT
from treelstm import TreeLSTM, TreeLSTMDouble, TreeLSTM_wo_MLP
from tqdm import tqdm
import pandas as pd
import numpy as np
from moco import MoCoTreeLSTM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from fast_pytorch_kmeans import KMeans
from sklearn.metrics import confusion_matrix
from collections import namedtuple
import dgl

all_wo_others = {
    "VPM": 0,
    "Isocortex_layer23": 1,
    "Isocortex_layer4": 2,
    "PRE": 3,
    "SUB": 4,
    "CP": 5,
    "VPL": 6,
    "Isocortex_layer6": 7,
    "MG": 8,
    "Isocortex_layer5": 9,
}
label_idx_to_class_names = {v: k for k, v in all_wo_others.items()}
NeuronBatch = namedtuple("NeuronBatch", ["graph", "feats", "label", "offset"])


def collate_fn(batch):
    trees, offsets, labels, height = [], [0], [], []
    n_nodes = []
    cnt = 0
    for b in batch:
        trees.append(b[0])
        cnt += b[1]
        n_nodes.append(b[1])
        offsets.append(cnt)
        labels.append(b[2])
    offsets.pop(-1)
    batch_trees = dgl.batch(trees)
    return NeuronBatch(
        graph=batch_trees.to("cuda"),
        feats=batch_trees.ndata["feats"].float().to("cuda"),
        label=torch.as_tensor(labels).to("cuda"),
        offset=torch.IntTensor(offsets).to("cuda"),
    )


class DebugNeuronTreeDataset(NeuronTreeDataset):
    def __init__(
        self,
        phase="full",
        input_features=[2, 3, 4],
        topology_transformations=None,
        attribute_transformations=None,
        dataset="all_wo_others",
        data_dir=None,
        label_dict=all_wo_others,
    ):
        assert phase in ["train", "test", "full"]
        super(DebugNeuronTreeDataset, self).__init__(
            phase=phase,
            input_features=input_features,
            topology_transformations=topology_transformations,
            attribute_transformations=attribute_transformations,
            dataset=dataset,
            data_dir=data_dir,
            label_dict=label_dict,
        )

    def __getitem__(self, idx):
        length = self.lines[idx][:, 3]
        contraction = self.lines[idx][:, 4]
        avg_length, std_length = np.mean(length), np.std(length)
        avg_contraction, std_contraction = np.mean(contraction), np.std(contraction)

        return (
            self.neuron_trees[idx],
            self.tree_lens[idx],
            self.targets[idx],
            self.tree_heights[idx],
            avg_length,
            std_length,
            avg_contraction,
            std_contraction,
        )


def plot_tsne_and_cm(df, feats, dataset, alpha=1.0, ncol=None):
    y_true = df["class"]
    tmp_df = df.copy()
    y_unq = np.unique(y_true)
    _feats = feats[tmp_df.index]
    centroids = torch.tensor([np.mean(_feats[y_true == l], axis=0) for l in y_unq])
    kmeans = KMeans(n_clusters=len(y_unq), mode="cosine")
    cluster_labels = kmeans.fit_predict(
        torch.tensor(_feats), centroids=centroids
    ).numpy()
    tmp_df["cluster_labels"] = cluster_labels
    _y_true = y_true.copy()
    for i, l in enumerate(y_unq):
        _y_true[y_true == l] = i
    cm = confusion_matrix(_y_true, cluster_labels)
    title = f"{dataset} Confusion Matrix"
    target_names = [label_idx_to_class_names[l] for l in y_unq]
    normalize = "true"

    cmap = plt.get_cmap("Blues")

    with np.errstate(all="ignore"):
        if normalize == "true":
            color_cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            color_cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            color_cm = cm / cm.sum()
        color_cm = np.nan_to_num(color_cm)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(color_cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:,}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] == np.max(cm[i, :]) else "black",
        )
    plt.ylabel("True label")

    ncol = len(y_unq) if ncol is None else ncol
    plt.subplot(1, 3, 2)
    sns.scatterplot(
        x=tmp_df["x"],
        y=tmp_df["y"],
        hue=tmp_df["cluster_labels"],
        palette=sns.color_palette("hls", len(y_unq)),
        alpha=alpha,
    )
    legend = plt.legend(
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=ncol,
        fancybox=True,
        title="cluster labels",
    )
    plt.title("clustered label", y=-0.2)
    plt.subplot(1, 3, 3)
    sns.scatterplot(
        x=tmp_df["x"],
        y=tmp_df["y"],
        hue=tmp_df["class_name"],
        palette=sns.color_palette("hls", len(y_unq)),
        alpha=alpha,
    )
    legend = plt.legend(
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=ncol,
        fancybox=True,
        title="class name",
    )
    plt.title("true label", y=-0.2)
    plt.tight_layout()
    plt.savefig("CM_all.pdf", dpi=200)
    tmp_df.to_csv("tmp.csv", index=False)


def plot_tsne_and_cm_v2(df, dataset, alpha=1.0, ncol=None, fname=None):
    y_true = df["class"]
    tmp_df = df.copy()
    y_unq = np.unique(y_true)
    cm = confusion_matrix(y_true, tmp_df["K=40"])
    cm = cm[: np.max(y_unq) + 1]
    cluster_labels = np.zeros(len(tmp_df))
    cluster_names = np.array([f"{i}" for i in range(len(tmp_df))], dtype=object)
    for i in range(40):
        try:
            l = tmp_df[tmp_df["K=40"] == i]["class"].mode()[0]
            cluster_labels[tmp_df[tmp_df["K=40"] == i].index] = l
            cluster_names[tmp_df[tmp_df["K=40"] == i].index] = label_idx_to_class_names[
                l
            ]
        except:
            import pdb

            pdb.set_trace()
            print("hi")

    tmp_df["cluster_labels"] = cluster_labels
    tmp_df["cluster_names"] = cluster_names
    cm_merged = confusion_matrix(y_true, tmp_df["cluster_labels"])
    target_names = [label_idx_to_class_names[l] for l in y_unq]
    normalize = "pred"

    cmap = plt.get_cmap("Blues")

    with np.errstate(all="ignore"):
        if normalize == "true":
            color_cm = cm / cm.sum(axis=1, keepdims=True)
            color_cm2 = cm_merged / cm_merged.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            color_cm = cm / cm.sum(axis=0, keepdims=True)
            color_cm2 = cm_merged / cm_merged.sum(axis=0, keepdims=True)
        elif normalize == "all":
            color_cm = cm / cm.sum()
            color_cm2 = cm_merged / cm_merged.sum()
        color_cm = np.nan_to_num(color_cm)
        color_cm2 = np.nan_to_num(color_cm2)

    plt.figure(figsize=(20, 10))
    # plt.figure(figsize=(15, 5))

    plt.subplot(2, 3, (1, 2))
    plt.imshow(color_cm, interpolation="nearest", cmap=cmap)
    plt.title(f"{dataset} Confusion Matrix (before merging)")
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(np.arange(40), np.arange(40), rotation=90)
        plt.yticks(tick_marks, target_names)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:,}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] == np.max(cm[:, j]) else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted")
    ncol = len(y_unq) if ncol is None else ncol
    plt.subplot(2, 3, 3)
    sns.scatterplot(
        x=tmp_df["x"],
        y=tmp_df["y"],
        hue=tmp_df["K=40"],
        palette=sns.color_palette("hls", 40),
        legend="brief",
        alpha=alpha,
    )
    legend = plt.legend(
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=8,
        fancybox=True,
        title="cluster name",
    )
    plt.title("clustered label", y=-0.2)
    ##############################################
    plt.subplot(2, 3, 4)
    # plt.subplot(1,3,1)
    plt.imshow(color_cm2, interpolation="nearest", cmap=cmap)
    plt.title(f"{dataset} Confusion Matrix (after merging)")
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    for i, j in itertools.product(range(cm_merged.shape[0]), range(cm_merged.shape[1])):
        plt.text(
            j,
            i,
            "{:,}".format(cm_merged[i, j]),
            horizontalalignment="center",
            color="white" if cm_merged[i, j] == np.max(cm_merged[:, j]) else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted")
    ncol = len(y_unq) if ncol is None else ncol
    # plt.subplot(1,3,2)

    # plt.subplot(1,3,3)
    plt.subplot(2, 3, 5)
    sns.scatterplot(
        x=tmp_df["x"],
        y=tmp_df["y"],
        hue=tmp_df["class_name"],
        palette=sns.color_palette("hls", len(y_unq)),
        alpha=alpha,
    )
    legend = plt.legend(
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=ncol,
        fancybox=True,
        title="class name",
    )
    plt.title("true label", y=-0.2)

    plt.subplot(2, 3, 6)
    print(np.unique(tmp_df["cluster_labels"]), len(np.unique(tmp_df["cluster_labels"])))
    print(np.unique(tmp_df["cluster_names"]), len(np.unique(tmp_df["cluster_names"])))
    sns.scatterplot(
        x=tmp_df["x"],
        y=tmp_df["y"],
        hue=tmp_df["cluster_names"],
        palette=sns.color_palette("hls", len(np.unique(tmp_df["cluster_names"]))),
        alpha=alpha,
    )
    legend = plt.legend(
        bbox_to_anchor=(0.5, 1.0),
        loc="lower center",
        ncol=ncol,
        fancybox=True,
        title="cluster name",
    )
    plt.title("clustered label", y=-0.2)

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    return tmp_df


def cluster(model, loader, batch0, K=[10, 20, 30, 40], n_nets=2):
    df = {
        "class": loader.dataset.targets,
        "class_name": [
            loader.dataset.label_idx_to_class_names[x] for x in loader.dataset.targets
        ],
        "height": loader.dataset.tree_heights,
        "n_nodes": loader.dataset.tree_lens,
        "avg_length": [
            np.mean(loader.dataset.lines[x][:, 3]) for x in range(len(loader.dataset))
        ],
        "total_length": [
            np.sum(loader.dataset.lines[x][:, 3]) for x in range(len(loader.dataset))
        ],
        "avg_contraction": [
            np.mean(loader.dataset.lines[x][:, 4]) for x in range(len(loader.dataset))
        ],
        "std_contraction": [
            np.std(loader.dataset.lines[x][:, 4]) for x in range(len(loader.dataset))
        ],
    }
    try:
        if n_nets == 2:
            features = [[], []]
            for i, net in enumerate([model.backbone_q, model.encoder_q]):
                net.eval()
                net(batch0)
                with torch.no_grad():
                    for batch in tqdm(loader, desc="Feature extracting"):
                        feature = net(batch)
                        features[i].append(feature)
                features[i] = torch.cat(features[i], dim=0).contiguous()
            features = torch.cat(features, dim=1)
        else:
            features = []
            net = model.backbone_q
            net.eval()
            net(batch0)
            with torch.no_grad():
                for batch in tqdm(loader, desc="Feature extracting"):
                    feature = net(batch)
                    features.append(feature)
                features = torch.cat(features, dim=0).contiguous()
    except:
        features = []
        model.eval()
        model.forward_backbone(batch0)
        with torch.no_grad():
            for batch in tqdm(loader, desc="Feature extracting"):
                feature = model.forward_backbone(batch)
                features.append(feature)
            features = torch.cat(features, dim=0).contiguous()
    features = nn.functional.normalize(features, dim=1)
    features = features.cpu().numpy()
    for k in K:
        kmeans = Kmeans(k=k, pca_dim=-1)
        kmeans.cluster(features, seed=2000)
        cluster_labels = kmeans.labels.astype(np.int64)
        df[f"K={k}"] = cluster_labels
    file_names = loader.dataset.file_list
    df["dataset"] = [x.split("/")[-2].split("-")[0] for x in file_names]
    tsne = TSNE(n_components=2, verbose=1, random_state=36)
    coords = tsne.fit_transform(features)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["file_name"] = file_names
    try:
        df = pd.DataFrame(data=df)
    except:
        import pdb

        pdb.set_trace()
        print("hi")
    # df.to_csv(out_file, index=False)
    return df, features


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Supervised Training")
    # Basic
    parser.add_argument("--dataset", type=str, default="all_wo_others")
    # Model
    parser.add_argument("--model", type=str, default="double")
    parser.add_argument("--clf", action="store_true", default=False)
    parser.add_argument("--child_mode", type=str, default="sum", help="[sum, average]")
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
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
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
    # others
    parser.add_argument(
        "--eval_part", type=str, default="full", help="[full | backbone]"
    )
    parser.add_argument(
        "--use_translation_feats",
        action="store_true",
        default=False,
        help="use 24-d translation feats",
    )
    parser.add_argument(
        "--pretrained", default="", type=str, help="path to pretrained checkpoint"
    )
    # work_dir/fullset/base_wt_others_unbalanced/epoch_40.pth

    args = parser.parse_args()
    args.bn = True
    args.projector_bn = True
    # args.use_translation_feats = False
    # args.input_features = [2,3,4]
    # args.pretrained = 'work_dir/clf/double_5_seu_6_classes_aug_all/epoch_100.pth'
    # args.pretrained = 'work_dir/final/double_5_bn_proj_bn_all/epoch_65.pth'
    args.pretrained = "work_dir/rebuttal/random_init_ori/epoch_100.pth"
    # args.pretrained = 'work_dir/arch_ablation/full_double_29_bn_proj_bn_all/epoch_90.pth'
    # args.pretrained = 'work_dir/arch_ablation/full_double_5_bn_proj_bn_all-drop_0_0_0.005/epoch_65.pth'
    # args.pretrained = 'work_dir/topo_ablation/double_dummy_coords_normal/epoch_75.pth'
    if args.use_translation_feats:
        args.input_features += [i for i in range(20, 44)]
    # hyper parameters
    device = torch.device("cuda")
    # create the model
    print(json.dumps(vars(args), indent=4, sort_keys=True))  # print args
    print("=> creating models ...")
    if not args.clf:
        model = MoCoTreeLSTM(
            args,  # for TreeLSTM
            dim=args.moco_dim,
            K=args.moco_k,
            m=args.moco_m,
            T=args.moco_t,
            symmetric=args.symmetric,
        ).to(device)
    else:
        model = TreeLSTMDouble(
            len(args.input_features),
            h_size=args.h_size,
            num_classes=len(LABEL_DICT["seu_6_classes"]),
        ).to(device)
    print(model)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.pretrained, map_location=loc)
            model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.pretrained, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    cudnn.benchmark = True

    dataset = DebugNeuronTreeDataset(
        phase="full",
        input_features=args.input_features,
        dataset=args.dataset,
        label_dict=LABEL_DICT[args.dataset],
    )

    loader = DataLoader(
        dataset=dataset, batch_size=512, collate_fn=collate_fn, shuffle=False
    )
    for batch in loader:
        batch0 = batch
        break
    # file_name = 'final_results/rebuttal/trnn_29_feats.csv'
    # feats_pth = 'final_results/rebuttal/trnn_29_feats.npy'
    # force_recompute = False
    # if os.path.exists(file_name) and not force_recompute:
    #     df = pd.read_csv(file_name)
    #     feats = np.load(feats_pth)
    # else:
    #     df, feats = cluster(model, loader, batch0, n_nets=1)
    #     np.save(feats_pth,feats)
    #     df.to_csv(file_name, index=False)
    # import matplotlib
    # plt.figure(figsize=(20,20))
    # cmap = matplotlib.cm.get_cmap('tab10')
    # for i in range(len(df)):
    #     plt.text(df.iloc[i]['x'], df.iloc[i]['y'], f'{i}', color=cmap(df.iloc[i]['class']))
    # # plt.text(x, y, s, fontsize=12)
    # # tmp_df = df[df['class']!=9]
    # # tmp_df = plot_tsne_and_cm_v2(df, 'ALL', ncol=3, alpha=0.8, fname='final_results/5_feats_all_backbone.pdf')

    # print(feats.shape)
    # sns.scatterplot(x=df['x'], y=df['y'], hue=df['class_name'])
    # legend = plt.legend(
    #                 bbox_to_anchor=(0.5, 1.0),
    #                 loc='lower center',
    #                 ncol=3,
    #                 fancybox=True,
    #                 title='class name')
    # plt.tight_layout()
    # plt.savefig('final_results/rebuttal/trnn_29_feats.pdf')
    df = pd.DataFrame()
    features = np.load(
        "vis_analysis/rebuttal/train_seu_6class_xyzlcV24_model_best.pt_all_latent.npy"
    )
    file_names = open(
        "vis_analysis/rebuttal/train_seu_6class_xyzlcV24_model_best.pt_all_filelist.txt"
    ).readlines()
    file_names = [x.strip() for x in file_names]
    tsne = TSNE(n_components=2, verbose=1, random_state=36)
    coords = tsne.fit_transform(features)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["file_name"] = file_names
    file_name = "final_results/rebuttal/morphoVAE_29_feats.csv"
    # import matplotlib
    # plt.figure(figsize=(20,20))
    # cmap = matplotlib.cm.get_cmap('tab10')
    # for i in range(len(df)):
    #     plt.text(df.iloc[i]['x'], df.iloc[i]['y'], f'{i}')
    # sns.scatterplot(x=df['x'], y=df['y'])
    # legend = plt.legend(
    #                 bbox_to_anchor=(0.5, 1.0),
    #                 loc='lower center',
    #                 ncol=3,
    #                 fancybox=True,
    #                 title='class name')
    df.to_csv(file_name, index=False)
    # plt.savefig('final_results/rebuttal/morphoVAE_29_feats.pdf')
