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

data_path = "../data"

# convert JML json file
convert_janelia_json(
    os.path.join(data_path, "raw/janelia_mouselight/json30/*.json"),
    os.path.join(data_path, "raw/janelia_mouselight/swc"),
    os.path.join(data_path, "info/JML_info_swc.csv"),
)

# preprocess data
for source in ["janelia_mouselight", "allen_cell_type", "seu_nature"]:
    print(f"Processing:{source}")
    # normalize neuron's center, orientation and size
    print(f"Normalize neuron")
    folder_in = f"{data_path}/raw/{source}/swc/"
    folder_out = f"{data_path}/raw/{source}/swc_soma0/"
    if source == "seu_nature":
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
