"""
Please refer to data/README.md for data download and preparation.
Then run this script to preprocess data.
"""

import os, sys
from data_io.process_raw_script import (
    convert_janelia_json,
    normalize_root_and_check,
    filter_axon_and_check,
    summarize_branch,
    split_sample_10fold_cv_and_merge,
)
import pandas as pd

data_path = "./data"

# convert JML json file
convert_janelia_json(
    os.path.join(data_path, "raw/janelia_mouselight/json30/*.json"),
    os.path.join(data_path, "raw/janelia_mouselight/swc"),
    os.path.join(data_path, "info/JML_info_swc.csv"),
)

# preprocess data
for source in ["janelia_mouselight", "allen_cell_type", "bil"]:
    print(f"Processing:{source}")
    # normalize neuron's center, orientation and size
    print(f"Normalize neuron")
    folder_in = f"{data_path}/raw/{source}/swc/"
    folder_out = f"{data_path}/raw/{source}/swc_soma0/"
    if source == "bil":
        normalize_root_and_check(folder_in + "*reg.swc", folder_out)
        # some BIL reconstructions are not correctly scaled. this will fix them
        normalize_root_and_check(
            folder_in + "*__reg.swc", folder_out, scale=[0.114, 0.114, 0.28]
        )
    else:
        normalize_root_and_check(folder_in + "*.swc", folder_out)

    # remove axon file
    print(f"Remove axons")
    folder_in = f"{data_path}/raw/{source}/swc_soma0/*.swc"
    folder_out = f"{data_path}/dendrite/{source}/swc_soma0/"
    filter_axon_and_check(folder_in, folder_out)

    print(f"Calculate features")
    folder_in = f"{data_path}/dendrite/{source}/swc_soma0/*.swc"
    folder_out = f"{data_path}/dendrite/{source}/eswc_soma0/"
    summarize_branch(folder_in, folder_out)


# split data into 10 folds
split_sample_10fold_cv_and_merge(data_path)

# hack folder creation
folder_names = ["allen_cell_type", "bil", "janelia_mouselight"]
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
for i, split_csv in enumerate(
    [
        f"{data_path}/info/ACT_info_swc_10folds.csv",
        f"{data_path}/info/BIL_info_swc_10folds.csv",
        f"{data_path}/info/JML_info_swc_10folds.csv",
    ]
):
    csv = pd.read_csv(split_csv)
    folder_name = folder_names[i]
    for split in range(10):
        for fname in csv[csv["model__fold"] == split]["swc__fname"]:
            # get acronym from "structure_merge__acronym"
            acronym = csv[csv["swc__fname"] == fname][
                "structure_merge__acronym"
            ].values[0]
            if acronym not in all_wo_others:
                if acronym == "Isocortex_layer2/3":
                    acronym = "Isocortex_layer23"
                else:
                    continue
            os.makedirs(
                f"{data_path}/dendrite/all_eswc_soma0_ssl/{acronym}/{folder_name}-{split}/",
                exist_ok=True,
            )
            source_path = os.path.abspath(
                f"{data_path}/dendrite/{folder_name}/eswc_soma0/{fname}"
            )
            target_path = f"{data_path}/dendrite/all_eswc_soma0_ssl/{acronym}/{folder_name}-{split}/{fname}"

            # Check if the source path is a valid file
            if os.path.isfile(source_path):
                os.symlink(source_path, target_path)
            else:
                print(f"{source_path} is not a valid file!")
