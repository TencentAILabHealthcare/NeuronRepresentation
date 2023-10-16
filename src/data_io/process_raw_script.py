import sys, os
from pathlib import Path

root_path = str(Path(os.path.abspath(__file__)).parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import glob
import pandas as pd
import json
import os
from collections import defaultdict
import random
from tqdm import tqdm
from allensdk.core.structure_tree import StructureTree

from summarize_branch import NeuronSimplifier
from sys_util import mkdir
from neuron import Neuron, Vertex


class MergeRegionHelper:
    def __init__(self):
        self.tree = get_allen_structure_tree()

    def get_merged_region_name(self, row):
        res = "unknown"
        if row["structure_parent__id"]:
            structure = self.tree.get_structures_by_id([row["structure_parent__id"]])[0]
            if not structure:
                return res
            if 315 in structure["structure_id_path"]:
                layer = row["structure__layer"]
                if layer in ("6a", "6b"):
                    layer = 6
                res = "Isocortex_layer" + str(layer)
            else:
                res = row["structure_parent__acronym"]
        return res


def get_allen_structure_tree():
    fname_tree = "./data/info/tree.json"
    with open(fname_tree) as fp:
        structure_graph = json.load(fp)
    tree = StructureTree(structure_graph)
    return tree


def convert_janelia_json(
    folder_in,
    folder_out,
    csv_out,
):
    """
    iterate folder and conver all janelia mouselight json file into swc files
    """
    filelist = glob.glob(folder_in)
    mkdir(folder_out)
    tree = get_allen_structure_tree()

    row = defaultdict(list)
    for fname_json in tqdm(filelist):
        fname_swc = os.path.join(
            folder_out, os.path.basename(fname_json).replace("json", "swc")
        )

        # converte file
        neuron = Neuron()
        neuron.load_json(fname_json)
        neuron.save_swc(fname_swc)

        # parse metainfo from json and save in csv
        with open(fname_json) as fp:
            data = json.load(fp)
        neuron = data["neuron"]
        row["specimen__id"].append(neuron["idString"])
        row["DOI"].append(neuron["DOI"])
        row["specimen__date"].append(neuron["sample"]["date"])
        row["specimen__strain"].append(neuron["sample"]["strain"])
        row["label__virus"].append(neuron["label"]["virus"])
        row["label__fluorophore"].append(
            neuron["label"]["fluorophore"].replace(",", ";")
        )
        row["swc__fname"].append(os.path.basename(fname_swc))

        if neuron["soma"]["allenId"] is not None:
            row["structure__id"].append(neuron["soma"]["allenId"])
            # use info from allen dicitonary
            structure = tree.get_structures_by_id([neuron["soma"]["allenId"]])[0]
            if structure is None:
                print(neuron["soma"]["allenId"])
                import pdb

                pdb.set_trace()
            row["structure__acronym"].append(structure["acronym"].replace(",", "/"))
            if "layer " in structure["name"]:
                row["structure__layer"].append(structure["name"].split("layer ")[1])
                parent = tree.get_structures_by_id(
                    [structure["structure_id_path"][-2]]
                )[0]
            else:
                row["structure__layer"].append(None)
                parent = structure
            row["structure_parent__id"].append(parent["id"])
            row["structure_parent__acronym"].append(parent["acronym"].replace(",", "/"))
        else:
            row["structure__id"].append(None)
            row["structure__acronym"].append(None)
            row["structure_parent__id"].append(None)
            row["structure_parent__acronym"].append(None)
            row["structure__layer"].append(None)

    df = pd.DataFrame(row)
    df.to_csv(csv_out)


def normalize_root_and_check(folder_in, folder_out, scale=None):
    """
    move root to origin and
    """
    if scale is None:
        scale = [1, 1, 1]
    mkdir(folder_out)
    filelist = glob.glob(folder_in)
    for fname in tqdm(filelist):
        neuron = Neuron()
        neuron.load_eswc(fname)
        if len(neuron.roots) != 1:
            print(fname, "has %d roots!" % len(neuron.roots))
            neuron.remove_fragements(flag_del_frag=True)
        neuron.scale_coordiante(scale[0], scale[1], scale[2])
        neuron.normalize_neuron()
        neuron.override_type()
        neuron.save_swc(os.path.join(folder_out, os.path.basename(fname)))


def filter_axon_and_check(folder_in, folder_out):
    """
    keep dendrite branch only
    """
    mkdir(folder_out)
    filelist = glob.glob(folder_in)
    for fname in tqdm(filelist):
        neuron = Neuron()
        neuron.load_eswc(fname)
        if len(neuron.roots) != 1:
            print(
                fname,
                "has %d roots! Will remove fragements and keep only 1 root."
                % len(neuron.roots),
            )
            neuron.remove_fragements(flag_del_frag=True)
        neuron.override_type()
        neuron.remove_branch_by_type([2])
        neuron.save_swc(os.path.join(folder_out, os.path.basename(fname)))


def summarize_branch(
    folder_in="/home/hanbo/data/dendrite/$source/swc_soma0/",
    folder_out="/home/hanbo/data/dendrite/$source/eswc_soma0/",
):
    """
    summarize and simply neuron branches
    """
    mkdir(folder_out)
    filelist = glob.glob(folder_in)
    for fname in tqdm(filelist):
        N = Neuron()
        N.load_eswc(fname)
        if len(N.roots) == 0:
            print("Empty file:", fname)
            continue
        NS = NeuronSimplifier(N)
        NS.summarize_by_branch(os.path.join(folder_out, os.path.basename(fname)))


def split_sample_10fold_cv_and_merge(folder_data):
    folder_info = os.path.join(folder_data, "info")
    sample_dict = defaultdict(
        list
    )  # key is group name, value is next availabel fold idx
    flist_csv = [
        os.path.join(folder_info, "ACT_info_swc.csv"),
        os.path.join(folder_info, "BIL_info_swc.csv"),
        os.path.join(folder_info, "JML_info_swc.csv"),
    ]
    merger = MergeRegionHelper()
    random.seed(44)

    df_list = []
    for fname_csv in flist_csv:
        df = pd.read_csv(fname_csv)
        for rid, row in df.iterrows():
            group = "%s_layer%s" % (
                str(row["structure_parent__acronym"]).replace(",", ""),
                str(row["structure__layer"]).replace("/", ""),
            )
            if len(sample_dict[group]) == 0:
                tmp_list = list(range(10))
                random.shuffle(tmp_list)
                sample_dict[group] = tmp_list
            df.loc[rid, "model__fold"] = sample_dict[group].pop()
            df.loc[rid, "structure__safe_name"] = group
            df.loc[rid, "structure_merge__acronym"] = merger.get_merged_region_name(row)
        df.to_csv(fname_csv[:-4] + "_10folds.csv")
        df_list.append(df)
    df_all = pd.concat(df_list)
    df_all.to_csv(os.path.join(folder_info, "info_swc_all_10folds.csv"))
