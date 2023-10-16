from logging import raiseExceptions
import torch, dgl, os, pickle
import networkx as nx
from dgl.convert import from_networkx
from collections import namedtuple
from nltk import tree
from tqdm import tqdm
import numpy as np
import re

NeuronBatch = namedtuple("NeuronBatch", ["graph", "feats", "label", "offset"])
NeuronBatchSingle = namedtuple("NeuronBatchSingle", ["graph", "feats", "offset"])
NeuronBatchTwoViews = namedtuple("NeuronBatchTwoViews", ["view1", "view2"])

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

BIL_6_classes = {
    "CP": 0,
    "Isocortex_layer23": 1,
    "Isocortex_layer4": 2,
    "Isocortex_layer5": 3,
    "Isocortex_layer6": 4,
    "VPM": 5,
}

JM_classes = {
    "Isocortex_layer23": 0,
    "Isocortex_layer5": 1,
    "Isocortex_layer6": 2,
    "VPM": 3,
}

ACT_classes = {
    "Isocortex_layer23": 0,
    "Isocortex_layer4": 1,
    "Isocortex_layer5": 2,
    "Isocortex_layer6": 3,
}

LABEL_DICT = {
    "all_wo_others": all_wo_others,
    "bil_6_classes": BIL_6_classes,
    "JM": JM_classes,
    "ACT": ACT_classes,
}


def get_collate_fn(device, use_two_views=False):
    def collate_fn(batch):
        trees, offsets, labels = [], [0], []
        cnt = 0
        for b in batch:
            trees.append(b[0])
            cnt += b[1]
            offsets.append(cnt)
            labels.append(b[2])
        offsets.pop(-1)
        batch_trees = dgl.batch(trees)
        return NeuronBatch(
            graph=batch_trees.to(device),
            feats=batch_trees.ndata["feats"].float().to(device),
            label=torch.as_tensor(labels).to(device),
            offset=torch.IntTensor(offsets).to(device),
        )

    def collate_fn_two_views(batch):
        trees1, offsets1, trees2, offsets2 = [], [0], [], [0]
        cnt1, cnt2 = 0, 0
        for b in batch:
            trees1.append(b[0])
            trees2.append(b[2])
            cnt1 += b[1]
            cnt2 += b[3]
            offsets1.append(cnt1)
            offsets2.append(cnt2)
        offsets1.pop(-1)
        offsets2.pop(-1)
        batch_trees1, batch_trees2 = dgl.batch(trees1), dgl.batch(trees2)
        return NeuronBatchTwoViews(
            view1=NeuronBatchSingle(
                graph=batch_trees1.to(device),
                feats=batch_trees1.ndata["feats"].float().to(device),
                offset=torch.IntTensor(offsets1).to(device),
            ),
            view2=NeuronBatchSingle(
                graph=batch_trees2.to(device),
                feats=batch_trees2.ndata["feats"].float().to(device),
                offset=torch.IntTensor(offsets2).to(device),
            ),
        )

    return collate_fn_two_views if use_two_views else collate_fn


def get_tree_len(tree):
    tree_len = 1
    for child in tree:
        tree_len += get_tree_len(child)
    return tree_len


class NeuronTreeDataset(object):
    def __init__(
        self,
        phase="train",
        input_features=[2, 3, 4, 12, 13],
        topology_transformations=None,
        attribute_transformations=None,
        dataset="bil_6_classes",
        data_dir=None,
        labels=None,
        label_dict=None,
        balance=True,
        reload=False,
    ):
        assert phase in ["train", "test", "full"]
        self.phase = phase
        self.input_features = input_features
        self.topology_transformations = topology_transformations
        self.attribute_transformations = attribute_transformations
        self.dataset = dataset
        self.data_dir = data_dir
        self.balance = balance
        if label_dict is None and labels is not None:
            self.classes = labels
            self.label_dict = {_class: i for i, _class in enumerate(self.classes)}
        elif label_dict is not None:
            self.label_dict = label_dict
            self.classes = label_dict.keys()
        else:
            raise ValueError("One of label_dict and labels should be given")
        self.label_idx_to_class_names = {v: k for k, v in self.label_dict.items()}
        self.reload = reload
        self.load_filecache()
        # print stats
        print(self.dataset, phase)
        print(
            self.dataset,
            phase,
            "distribution:",
            np.unique(self.targets, return_counts=True),
        )
        print(self.dataset, phase, self.label_dict)

        # faster caching for no augmentation dataset
        if topology_transformations is None and attribute_transformations is None:
            self.neuron_trees, self.tree_lens, self.tree_heights = self.process()
        else:
            self.neuron_trees, self.tree_lens, self.tree_heights = None, None, None

    def process(self):
        neurons = []
        tree_heights = []
        for i, line in enumerate(self.lines.copy()):
            root = self.lines2trees(line, self.i2ps[i])
            neurons.append(root)
            tree_heights.append(root.height())
        neuron_trees = []
        tree_lens = []
        for neuron in neurons:
            _tree, tree_len = self._build_tree(neuron)
            neuron_trees.append(_tree)
            tree_lens.append(tree_len)
        return neuron_trees, tree_lens, tree_heights

    def load_filecache(self):
        if self.dataset != "rebuttal":
            if not os.path.exists("data/dendrite/processed_datasets/"):
                os.makedirs("data/dendrite/processed_datasets/")
            saved_cache = (
                f"data/dendrite/processed_datasets/{self.dataset}_{self.phase}.pkl"
            )
            print("loading from", saved_cache)
            if os.path.exists(saved_cache) and not self.reload:
                with open(saved_cache, "rb") as f:
                    saved_cache = pickle.load(f)
                lines, i2ps, targets = (
                    saved_cache["lines"],
                    saved_cache["i2ps"],
                    saved_cache["targets"],
                )
                file_list = saved_cache["file_list"]
            else:
                lines, i2ps, targets, file_list = self.load_and_cache(saved_cache)
            self.lines = [x[:, self.input_features] for x in lines]
            self.i2ps, self.targets = i2ps, targets
            self.file_list = file_list
        else:
            self.lines = []
            self.i2ps, self.targets, self.file_list = [], [], []
            for dataset in ["bil_6_classes", "JM", "ACT"]:
                saved_cache = f"data/dendrite/processed_datasets/{dataset}_train.pkl"
                print("loading from", saved_cache)
                with open(saved_cache, "rb") as f:
                    saved_cache = pickle.load(f)
                print(dataset, saved_cache.keys())
                lines, i2ps, targets = (
                    saved_cache["lines"],
                    saved_cache["i2ps"],
                    saved_cache["targets"],
                )
                # lines, i2ps, targets, file_list = self.load_and_cache(saved_cache)
                file_list = saved_cache["file_list"]
                lines = [x[:, self.input_features] for x in lines]
                self.lines.extend(lines)
                self.i2ps.extend(i2ps)
                self.targets.extend(targets)
                self.file_list.extend(file_list)
            idxs = np.arange(len(self.lines))
            np.random.shuffle(idxs)
            print(idxs)
            self.lines = np.array(self.lines)[idxs]
            self.i2ps = np.array(self.i2ps)[idxs]
            self.targets = np.array(self.targets)[idxs]
            self.file_list = np.array(self.file_list)[idxs]
        print(len(self.lines))
        print("dataset loaded")

    def load_and_cache(self, saved_cache):
        _file_list = self.list_file()
        cached = {"file_cache": {}}
        file_list, lines, i2ps, targets = [], [], [], []
        for sub_list in _file_list:
            for filename in sub_list:
                if filename not in cached["file_cache"]:
                    _lines = self.readswc(filename)
                    cached["file_cache"][filename] = _lines
                else:
                    _lines = cached["file_cache"][filename]
                file_list.append(filename)
                lines.append(cached["file_cache"][filename])
                i2ps.append(dict([[int(p[0]), int(p[6])] for p in _lines]))
                try:
                    targets.append(self.label_dict[filename.split("/")[-3]])
                except:
                    # for others
                    targets.append(self.label_dict[filename.split("/")[-4]])
        cached["lines"], cached["i2ps"], cached["targets"] = lines, i2ps, targets
        cached["file_list"] = file_list
        with open(saved_cache, "wb") as f:
            pickle.dump(cached, f)
        print("saving to", saved_cache)
        return lines, i2ps, targets, file_list

    def lines2trees(self, lines, i2p):
        p2t = dict()
        for i in i2p:
            if int(i2p[i]) == -1:
                root = tree.Tree(lines[int(i - 1)], [])
                p2t[i] = root
                break

        def add_node(i, line):
            temp = tree.Tree(line, [])
            if int(i2p[i]) not in p2t:
                add_node(int(i2p[i]), lines[int(i2p[i])])
            p2t[int(i2p[i])].append(temp)
            p2t[i] = temp

        for i, line in enumerate(lines):
            try:
                if int(i2p[i + 1]) == -1:
                    continue
            except KeyError:
                print(lines)
            if i + 1 not in p2t:
                add_node(i + 1, line)
        return root

    def _build_tree(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()
                g.add_node(cid, feats=child.label())
                if len(child) != 0:
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        g.add_node(0, feats=root.label())
        _rec_build(0, root)
        ret = from_networkx(g, node_attrs=["feats"])
        return ret, ret.number_of_nodes()

    def __getitem__(self, idx):
        if (
            self.attribute_transformations is None
            and self.topology_transformations is None
        ):
            return self.neuron_trees[idx], self.tree_lens[idx], self.targets[idx]
        else:
            lines = self.lines[idx].copy()

            if self.attribute_transformations is not None:
                lines = self.attribute_transformations(lines)

            neuron = self.lines2trees(lines, self.i2ps[idx])

            if self.topology_transformations is not None:
                neuron = self.topology_transformations(neuron)

            neuron_tree, tree_len = self._build_tree(neuron)
            label = self.targets[idx]
            return neuron_tree, tree_len, label

    def __len__(self):
        r"""Number of graphs in the dataset."""
        return len(self.targets)

    def list_file(self):
        assert self.data_dir is not None, "To list_file, you have to specifiy data_dir"
        self.data_dir = self.data_dir[self.phase]
        ls = [[] for d in self.data_dir]
        file_list = [[] for d in self.data_dir]

        for i in range(len(ls)):
            self.recur_listdir(self.data_dir[i], ls[i])

        print("reading data list ... ")
        all_lines = []
        for i, d in enumerate(ls):
            for filename in tqdm(d):
                if filename[-3:] != "swc":
                    continue
                with open(filename, "r") as f:
                    lines = f.read().strip().split("\n")

                j = 0
                while lines[j].startswith("#"):
                    j += 1
                lines = lines[j:]

                if len(lines) >= 10:
                    all_lines.append(len(lines))
                    file_list[i].append(filename)
        print(np.max(all_lines), np.min(all_lines), np.mean(all_lines))
        if self.balance:
            if self.phase == "train":
                m = max([len(fi) for fi in file_list])
                for fi in file_list:
                    if len(fi) < m:
                        try:
                            extra = np.random.choice(fi, size=m - len(fi))
                        except:
                            print(m, len(fi), m - len(fi))
                            continue
                        fi.extend(extra)

        print(self.dataset, ":", self.phase, [len(fi) for fi in file_list])

        return file_list

    def recur_listdir(self, path, dir_list):
        for f in os.listdir(path):
            if os.path.isdir(path + "/" + f):
                self.recur_listdir(path + "/" + f, dir_list)
            else:
                dir_list.append(path + "/" + f)

    def readswc(self, filename):
        with open(filename, "r") as f:
            lines = f.read().strip().split("\n")
        i = 0
        while lines[i].startswith("#"):
            i += 1
        lines = lines[i:]

        for i in range(len(lines)):
            lines[i] = re.split(r"[\[\],\s]", lines[i])
            while "" in lines[i]:
                lines[i].remove("")
        lines = np.array(lines).astype(float)
        # if self.filter_type:
        # lines = self.filter_lines(lines)
        return lines

    def filter_lines(self, datum):
        # unused
        datum = datum[~np.isin(datum[:, 1], self.filter_type), :]
        return datum


class NeuronTreeDatasetTwoViews(NeuronTreeDataset):
    def __init__(
        self,
        phase="train",
        input_features=[2, 3, 4, 12, 13],
        topology_transformations=None,
        attribute_transformations=None,
        dataset="data_bil_3class",
        data_dir=None,
        labels=None,
        label_dict=None,
    ):
        assert phase in ["train", "test", "full"]
        super(NeuronTreeDatasetTwoViews, self).__init__(
            phase=phase,
            input_features=input_features,
            topology_transformations=topology_transformations,
            attribute_transformations=attribute_transformations,
            dataset=dataset,
            data_dir=data_dir,
            labels=labels,
            label_dict=label_dict,
        )

    def __getitem__(self, idx):
        if (
            self.attribute_transformations is None
            and self.topology_transformations is None
        ):
            neuron_tree1 = self.neuron_trees[idx]
            tree_len1 = self.tree_lens[idx]
            neuron_tree2 = self.neuron_trees[idx]
            tree_len2 = self.tree_lens[idx]
            return neuron_tree1, tree_len1, neuron_tree2, tree_len2
        else:
            lines = self.lines[idx].copy()
            if self.attribute_transformations is not None:
                lines1 = self.attribute_transformations(lines.copy())
                lines2 = self.attribute_transformations(lines.copy())
            else:
                lines1 = lines2 = lines.copy()
            neuron1 = self.lines2trees(lines1, self.i2ps[idx])
            neuron2 = self.lines2trees(lines2, self.i2ps[idx])
            if self.topology_transformations is not None:
                neuron1 = self.topology_transformations(neuron1)
                neuron2 = self.topology_transformations(neuron2)
            neuron_tree1, tree_len1 = self._build_tree(neuron1)
            neuron_tree2, tree_len2 = self._build_tree(neuron2)
            return neuron_tree1, tree_len1, neuron_tree2, tree_len2


def process_diff_datasets():
    root = "data/dendrite/all_eswc_soma0_ssl"
    bil_classes = [
        "CP",
        "Isocortex_layer23",
        "Isocortex_layer4",
        "Isocortex_layer5",
        "Isocortex_layer6",
        "VPM",
    ]
    bil_train, bil_test, bil_full = [], [], []
    for label in bil_classes:
        bil_train += [f"{root}/{label}/bil-{i}" for i in range(8)]
        bil_test += [f"{root}/{label}/bil-{i}" for i in range(8, 10)]
        bil_full += [f"{root}/{label}/bil-{i}" for i in range(10)]
    bil_6_classes = {"train": bil_train, "test": bil_test, "full": bil_full}

    JM_classes = ["Isocortex_layer23", "Isocortex_layer5", "Isocortex_layer6", "VPM"]
    JMclasses_label_dict = {
        "Isocortex_layer23": 1,
        "Isocortex_layer5": 3,
        "Isocortex_layer6": 4,
        "VPM": 5,
    }
    ACT_classes = [
        "Isocortex_layer23",
        "Isocortex_layer5",
        "Isocortex_layer6",
        "Isocortex_layer4",
    ]
    ACT_classes_label_dict = {
        "Isocortex_layer23": 1,
        "Isocortex_layer5": 3,
        "Isocortex_layer6": 4,
        "Isocortex_layer4": 2,
    }
    JM_test = []
    for label in JM_classes:
        JM_test += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(10)]
    JM_4_classes = {"test": JM_test}

    ACT_test = []
    for label in ACT_classes:
        ACT_test += [f"{root}/{label}/allen_cell_type-{i}" for i in range(10)]

    ACT_4_classes = {"test": ACT_test}

    labels = [bil_classes, JM_classes, ACT_classes]
    label_dicts = [None, JMclasses_label_dict, ACT_classes_label_dict]
    dataset_names = ["bil_6_classes", "JM_4_classes", "ACT_4_classes"]
    for ix, data_dir in enumerate([bil_6_classes, JM_4_classes, ACT_4_classes]):
        for phase in ["train", "test", "full"]:
            if phase not in data_dir:
                continue
            ds = NeuronTreeDataset(
                phase=phase,
                input_features=[2, 3, 4, 12, 13],
                topology_transformations=None,
                attribute_transformations=None,
                dataset=dataset_names[ix],
                labels=labels[ix],
                label_dict=label_dicts[ix],
                data_dir=data_dir,
                reload=True,
            )


def get_all_datasets():
    root = "data/dendrite/all_eswc_soma0_ssl"
    classes = os.listdir(root)
    full_wt_others, full_wo_others = [], []
    for label in classes:
        if label == "others":
            subclasses = os.listdir(f"{root}/{label}/")
            for fine_label in subclasses:
                candidate_folds = os.listdir(f"{root}/{label}/{fine_label}")
                full_wt_others += [
                    f"{root}/{label}/{fine_label}/{x}" for x in candidate_folds
                ]
        else:
            candidate_folds = os.listdir(f"{root}/{label}")
            full_wt_others += [f"{root}/{label}/{x}" for x in candidate_folds]
            full_wo_others += [f"{root}/{label}/{x}" for x in candidate_folds]
    dataset_wt_others = {"full": full_wt_others}
    dataset_wo_others = {"full": full_wo_others}
    datasets = [dataset_wo_others]

    label_dicts = [all_wo_others]
    dataset_names = ["all_wo_others"]
    for ix, data_dir in enumerate(datasets):
        for phase in ["full"]:
            ds = NeuronTreeDataset(
                phase=phase,
                input_features=[2, 3, 4, 12, 13],
                topology_transformations=None,
                attribute_transformations=None,
                dataset=dataset_names[ix],
                label_dict=label_dicts[ix],
                data_dir=data_dir,
                reload=True,
            )
            print(len(ds.file_list))


def split_BIL_JM_ACT():
    root = "data/dendrite/all_eswc_soma0_ssl"

    bil_train, bil_test, bil_full = [], [], []
    for label in BIL_6_classes.keys():
        bil_train += [f"{root}/{label}/bil-{i}" for i in range(8)]
        bil_test += [f"{root}/{label}/bil-{i}" for i in range(8, 10)]
        bil_full += [f"{root}/{label}/bil-{i}" for i in range(10)]
    bil_6_classes = {"train": bil_train, "test": bil_test, "full": bil_full}

    JM_train, JM_test, JM_full = [], [], []
    for label in JM_classes.keys():
        JM_train += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(8)]
        JM_test += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(8, 10)]
        JM_full += [f"{root}/{label}/janelia_mouselight-{i}" for i in range(10)]
    JM_4_classes = {"train": JM_train, "test": JM_test, "full": JM_full}

    ACT_train, ACT_test, ACT_full = [], [], []
    for label in ACT_classes.keys():
        ACT_train += [f"{root}/{label}/allen_cell_type-{i}" for i in range(8)]
        ACT_test += [f"{root}/{label}/allen_cell_type-{i}" for i in range(8, 10)]
        ACT_full += [f"{root}/{label}/allen_cell_type-{i}" for i in range(10)]
    ACT_4_classes = {"train": ACT_train, "test": ACT_test, "full": ACT_full}

    label_dicts = [BIL_6_classes, JM_classes, ACT_classes]
    dataset_names = ["bil_6_classes", "JM", "ACT"]
    for ix, data_dir in enumerate([bil_6_classes, JM_4_classes, ACT_4_classes]):
        for phase in ["full", "train", "test"]:
            if phase not in data_dir:
                continue
            ds = NeuronTreeDataset(
                phase=phase,
                input_features=[2, 3, 4, 12, 13],
                topology_transformations=None,
                attribute_transformations=None,
                dataset=dataset_names[ix],
                label_dict=label_dicts[ix],
                data_dir=data_dir,
                reload=True,
                balance=False,
            )


if __name__ == "__main__":
    split_BIL_JM_ACT()
    get_all_datasets()
