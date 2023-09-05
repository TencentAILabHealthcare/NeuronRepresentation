# -*- coding: utf-8 -*-
"""
Created on Fri May 13 21:02:27 2022

@author: hanbochen
"""
import os
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from tqdm import tqdm
import pickle as pkl
from sklearn.manifold import TSNE


def plot_img(pdf, img_list, title):
    """
    save images in img_list to pdf file for visualization purpose
    img_list: [(img, color_map),(img2, color_map),...]
    """
    fig = plt.figure(figsize=(24 * len(img_list), 26))
    plt.title(title)
    plt.axis("off")
    subplot = 100 + len(img_list) * 10
    for idx in range(len(img_list)):
        ax = fig.add_subplot(subplot + idx + 1)
        if img_list[idx][1]:
            ax.imshow(img_list[idx][0], img_list[idx][1])
        else:
            ax.imshow(img_list[idx][0])
    pdf.savefig()
    plt.close()


def plot_clusters_pdf(fname_output, img_list):
    """
    save images in img_list to pdf file for visualization purpose
    img_list: [[(c1_img, color_map),(c1_img2, color_map),...],
               [(c2_img, color_map),(c2_img2, color_map),...],
               ...]
    """
    psize = 8
    pdf = PdfPages(fname_output)
    fig, axs = plt.subplots(
        len(img_list),
        len(img_list[0]),
        figsize=(psize * len(img_list[0]), psize * len(img_list)),
    )

    for idx_r in range(len(img_list)):
        for idx_c in range(len(img_list[idx_r])):
            ax = axs[idx_r][idx_c]
            ax.axis("off")
            if img_list[idx_r][idx_c][1]:
                ax.imshow(img_list[idx_r][idx_c][0], img_list[idx_r][idx_c][1])
            else:
                ax.imshow(img_list[idx_r][idx_c][0])
    pdf.savefig()
    plt.close()
    pdf.close()


def plot_clusters_img(fname_output, img_list, img_size=1000):
    """
    save images in img_list to pdf file for visualization purpose
    img_list: [[c1_img,c1_img2,...],
               [c2_img,c2_img2,...],
               ...]
    """
    padding = 185
    step = img_size - padding * 2
    img = np.zeros(
        (padding * 2 + step * len(img_list), padding * 2 + step * len(img_list[0]), 3)
    )

    for idx_r in range(len(img_list)):
        for idx_c in range(len(img_list[idx_r])):
            r = idx_r * step
            c = idx_c * step
            sub_img = cv2.imread(img_list[idx_r][idx_c][0])
            sub_img = sub_img * img_list[idx_r][idx_c][1]
            patch = img[r : (r + img_size), c : (c + img_size), :]
            patch[sub_img > 0] = sub_img[sub_img > 0]
            img[r : (r + img_size), c : (c + img_size), :] = patch
    cv2.imwrite(fname_output, 255 - img)


def plot_tsne_img(fname_output, img_list, r_list, c_list, cache=None, task_name="task"):
    """
    save images in img_list to pdf file for visualization purpose
    img_list: [[c1_img,c1_img2,...],
               [c2_img,c2_img2,...],
               ...]
    """
    r_left, r_right, c_left, c_right = 0, 0, 0, 0
    img_shape = 1000
    for img_info, r_center, c_center in zip(img_list, r_list, c_list):
        # sub_img = cv2.imread(img_info[0])
        # sub_img = sub_img * img_info[1]
        r_left = min(r_center - img_shape // 2 - 1, r_left)
        c_left = min(c_center - img_shape // 2 - 1, c_left)
        r_right = max(r_center + img_shape // 2 + 1, r_right)
        c_right = max(c_center + img_shape // 2 + 1, c_right)
    img = np.zeros((int(r_right - r_left + 1), int(c_right - c_left + 1), 3))
    for img_info, r_center, c_center in tqdm(
        zip(img_list, r_list, c_list), total=len(img_list), desc=task_name
    ):
        if cache is None:
            sub_img = cv2.imread(img_info[0])
        else:
            sub_img = cache[img_info[0]]
        # sub_img = 255-sub_img
        # sub_img[sub_img>0] = 255
        sub_img = sub_img * img_info[1]
        r = int(r_center - sub_img.shape[0] // 2 - r_left)
        c = int(c_center - sub_img.shape[1] // 2 - c_left)
        patch = img[r : (r + sub_img.shape[0]), c : (c + sub_img.shape[1]), :]
        patch[sub_img > 0] = sub_img[sub_img > 0]
        img[r : (r + sub_img.shape[0]), c : (c + sub_img.shape[1]), :] = patch
    cv2.imwrite(fname_output, 255 - img)
    del img


### scripts


def script_plot_tsne_graphcl():
    root_path = "neuron_morpho_dendrite"
    features = np.load("rebuttal/graphcl.npy")
    scale_coord = 100
    sample_frac = 1
    file_cache = pkl.load(
        open(
            "/apdcephfs/private_jiaweiyyang/project/neuron/NeuronRepresentation/datasets/neuron_morpho/processed_datasets_bk2/all_wo_others_full.pkl",
            "rb",
        )
    )
    file_lists = file_cache["file_list"]
    print(file_lists[0])
    coords = TSNE(n_components=2, verbose=1, random_state=99).fit_transform(features)
    df = pd.DataFrame(dict(x=coords[:, 0], y=coords[:, 1], file_name=file_lists))
    col_c = "x"
    col_r = "y"
    col_file = "file_name"

    df = df.sample(frac=sample_frac)
    np.random.seed(2000)
    attributes = ["random"]

    dataset_dict = {"seu_nature": 0, "janelia_mouselight": 1, "allen_cell_type": 2}
    k = 1
    views = ["xy"]
    for view in views:
        try:
            print("loading cache")
            with open("swc_screenshot_%s.pkl" % view, "rb") as handle:
                cache = pkl.load(handle)
            print("cache loaded")
        except:
            cache = None
        for attribute in attributes:
            if attribute == "dataset" or attribute == "class":
                cmap = matplotlib.cm.get_cmap("tab10")
            else:
                cmap = matplotlib.cm.get_cmap("jet")

            # fname_output = 'CL_tsne_all_color_%s'%attribute
            fname_output = "graphcl_all_color_%s" % attribute
            # fname_output = 'TRNN_all_color_%s'%attribute
            img_list = []
            r_list = []
            c_list = []
            for _, row in df.iterrows():
                _ds = row["file_name"].split("/")[-2].split("-")[0]
                fname_img = os.path.join(
                    root_path,
                    # row[col_dataset],
                    _ds,
                    # 'input_neuron',
                    "swc_soma0_screenshot",
                    os.path.basename(row[col_file]) + "_%s.png" % view,
                )
                # sub_img.append((255-img[350:650,350:650,:],None))
                if attribute == "dataset":
                    color = cmap(dataset_dict[row[attribute]])[:3]
                elif attribute in [
                    "class",
                    "height",
                    "n_nodes",
                    "avg_length",
                    "total_length",
                    "avg_contraction",
                    "std_contraction",
                ]:
                    color = cmap(row[attribute])[:3]
                else:
                    ## random color
                    color = np.random.random((1, 1, 3))
                    color /= np.max(color)  # make it dark
                # color -= np.min(color) # make it bright
                # color /= np.max(color) # make it dark
                img_list.append((fname_img, color))
                r_list.append(row[col_r] * scale_coord)
                c_list.append(row[col_c] * scale_coord)
            plot_tsne_img(
                fname_output + "_%s.png" % view,
                img_list,
                r_list,
                c_list,
                cache,
                task_name=f"{attribute}_{view} ({k}/{len(views)*len(attributes)})",
            )
            k += 1
        del cache


script_plot_tsne_graphcl()


def script_plot_tsne_vae():
    root_path = "neuron_morpho_dendrite"
    latents = np.load("rebuttal/train_all_xyzlcV24_ba_model_e99.pt_all_latent.npy")
    scale_coord = 100
    sample_frac = 1
    file_cache = pkl.load(
        open(
            "/apdcephfs/private_jiaweiyyang/project/neuron/NeuronRepresentation/datasets/neuron_morpho/processed_datasets_bk2/all_wo_others_full.pkl",
            "rb",
        )
    )
    file_lists = file_cache["file_list"]
    file_lists = [x.split("/")[-1] for x in file_lists]
    vae_filelist = open(
        "rebuttal/train_seu_6class_xyzlcV24_model_best.pt_all_filelist.txt"
    ).readlines()
    vae_ids = [x.split("/")[-1].strip() for x in vae_filelist]
    features, file_names = [], []
    for ix, id in enumerate(vae_ids):
        if id in file_lists:
            features.append(latents[ix])
            file_names.append(vae_filelist[ix].strip())
    coords = TSNE(n_components=2, verbose=1, random_state=36).fit_transform(features)
    df = pd.DataFrame(dict(x=coords[:, 0], y=coords[:, 1], file_name=file_names))
    col_c = "x"
    col_r = "y"
    col_file = "file_name"
    col_dataset = "dataset"

    df = df.sample(frac=sample_frac)
    np.random.seed(2000)
    attributes = ["random"]

    dataset_dict = {"seu_nature": 0, "janelia_mouselight": 1, "allen_cell_type": 2}
    k = 1
    views = ["xy"]
    for view in views:
        try:
            print("loading cache")
            with open("swc_screenshot_%s.pkl" % view, "rb") as handle:
                cache = pkl.load(handle)
            print("cache loaded")
        except:
            cache = None
        for attribute in attributes:
            if attribute == "dataset" or attribute == "class":
                cmap = matplotlib.cm.get_cmap("tab10")
            else:
                cmap = matplotlib.cm.get_cmap("jet")

            # fname_output = 'CL_tsne_all_color_%s'%attribute
            fname_output = "MorphoVAE_unsup_all_color_%s" % attribute
            # fname_output = 'TRNN_all_color_%s'%attribute
            img_list = []
            r_list = []
            c_list = []
            for _, row in df.iterrows():
                _ds = row["file_name"].split("/")[5]
                fname_img = os.path.join(
                    root_path,
                    # row[col_dataset],
                    _ds,
                    # 'input_neuron',
                    "swc_soma0_screenshot",
                    os.path.basename(row[col_file]) + "_%s.png" % view,
                )
                # sub_img.append((255-img[350:650,350:650,:],None))
                if attribute == "dataset":
                    color = cmap(dataset_dict[row[attribute]])[:3]
                elif attribute in [
                    "class",
                    "height",
                    "n_nodes",
                    "avg_length",
                    "total_length",
                    "avg_contraction",
                    "std_contraction",
                ]:
                    color = cmap(row[attribute])[:3]
                else:
                    ## random color
                    color = np.random.random((1, 1, 3))
                    color /= np.max(color)  # make it dark
                # color -= np.min(color) # make it bright
                # color /= np.max(color) # make it dark
                img_list.append((fname_img, color))
                r_list.append(row[col_r] * scale_coord)
                c_list.append(row[col_c] * scale_coord)
            plot_tsne_img(
                fname_output + "_%s.png" % view,
                img_list,
                r_list,
                c_list,
                cache,
                task_name=f"{attribute}_{view} ({k}/{len(views)*len(attributes)})",
            )
            k += 1
        del cache


# script_plot_tsne_vae()


def script_plot_tsne():
    root_path = "neuron_morpho_dendrite"
    latents = np.load("rebuttal/train_all_xyzlcV24_ba_model_e99.pt_all_latent.npy")
    scale_coord = 120
    sample_frac = 1
    file_cache = pkl.load(
        open(
            "/apdcephfs/private_jiaweiyyang/project/neuron/NeuronRepresentation/datasets/neuron_morpho/processed_datasets_bk2/all_wo_others_full.pkl",
            "rb",
        )
    )
    file_lists = file_cache["file_list"]
    file_lists = [x.split("/")[-1] for x in file_lists]
    vae_filelist = open(
        "rebuttal/train_seu_6class_xyzlcV24_model_best.pt_all_filelist.txt"
    ).readlines()
    vae_ids = [x.split("/")[-1].strip() for x in vae_filelist]
    features, file_names = [], []
    for ix, id in enumerate(vae_ids):
        if id in file_lists:
            features.append(latents[ix])
            file_names.append(vae_filelist[ix].strip())
    coords = TSNE(n_components=2, verbose=1, random_state=36).fit_transform(features)
    df = pd.DataFrame(dict(x=coords[:, 0], y=coords[:, 1], file_name=file_names))
    col_c = "x"
    col_r = "y"
    col_file = "file_name"
    col_dataset = "dataset"

    # df = df[df['class_name']!='Isocortex_layer5']
    df = df.sample(frac=sample_frac)
    # _ = df['class']
    # df['class'] = (_-_.min())/(_.max()-_.min())
    # _ = df['height']
    # df['height'] = (_-_.min())/(_.max()-_.min())
    np.random.seed(2000)
    # attributes = ['height', 'n_nodes', 'avg_length', 'total_length', 'avg_contraction', 'std_contraction']
    # for attribute in attributes:
    #     df[attribute] = (df[attribute]-df[attribute].min())/(df[attribute].max()-df[attribute].min())
    # attributes.insert(0, 'class')
    # attributes.insert(1, 'dataset')
    # attributes.insert(0, 'random')
    attributes = ["random"]

    dataset_dict = {"seu_nature": 0, "janelia_mouselight": 1, "allen_cell_type": 2}
    k = 1
    views = ["xy"]
    for view in views:
        try:
            print("loading cache")
            with open("swc_screenshot_%s.pkl" % view, "rb") as handle:
                cache = pkl.load(handle)
            print("cache loaded")
        except:
            cache = None
        for attribute in attributes:
            if attribute == "dataset" or attribute == "class":
                cmap = matplotlib.cm.get_cmap("tab10")
            else:
                cmap = matplotlib.cm.get_cmap("jet")

            # fname_output = 'CL_tsne_all_color_%s'%attribute
            fname_output = "MorphoVAE_unsup_all_color_%s" % attribute
            # fname_output = 'TRNN_all_color_%s'%attribute
            img_list = []
            r_list = []
            c_list = []
            for _, row in df.iterrows():
                _ds = row["file_name"].split("/")[5]
                fname_img = os.path.join(
                    root_path,
                    # row[col_dataset],
                    _ds,
                    # 'input_neuron',
                    "swc_soma0_screenshot",
                    os.path.basename(row[col_file]) + "_%s.png" % view,
                )
                # sub_img.append((255-img[350:650,350:650,:],None))
                if attribute == "dataset":
                    color = cmap(dataset_dict[row[attribute]])[:3]
                elif attribute in [
                    "class",
                    "height",
                    "n_nodes",
                    "avg_length",
                    "total_length",
                    "avg_contraction",
                    "std_contraction",
                ]:
                    color = cmap(row[attribute])[:3]
                else:
                    ## random color
                    color = np.random.random((1, 1, 3))
                    color /= np.max(color)  # make it dark
                # color -= np.min(color) # make it bright
                # color /= np.max(color) # make it dark
                img_list.append((fname_img, color))
                r_list.append(row[col_r] * scale_coord)
                c_list.append(row[col_c] * scale_coord)
            plot_tsne_img(
                fname_output + "_%s.png" % view,
                img_list,
                r_list,
                c_list,
                cache,
                task_name=f"{attribute}_{view} ({k}/{len(views)*len(attributes)})",
            )
            k += 1
        del cache


# script_plot_tsne()


def script_plot_tsne2():
    root_path = "neuron_morpho_dendrite"

    fname_csv = os.path.join(root_path, "results/5_feats_backbone.csv")
    scale_coord = 125
    sample_frac = 1

    ori_df = pd.read_csv(fname_csv)
    col_c = "x"
    col_r = "y"
    col_file = "file_name"
    col_dataset = "dataset"

    attributes = [
        "height",
        "n_nodes",
        "avg_length",
        "total_length",
        "avg_contraction",
        "std_contraction",
    ]
    for attribute in attributes:
        ori_df[attribute] = (ori_df[attribute] - ori_df[attribute].min()) / (
            ori_df[attribute].max() - ori_df[attribute].min()
        )
    attributes.insert(0, "class")
    attributes.insert(1, "dataset")
    # attributes.insert(0, 'K=20')
    # attributes.insert(0, 'random')
    np.random.seed(2000)

    dataset_dict = {"seu_nature": 0, "janelia_mouselight": 1, "allen_cell_type": 2}
    k = 1
    views = ["xy"]
    for view in views:
        try:
            print("loading cache")
            with open("swc_screenshot_%s.pkl" % view, "rb") as handle:
                cache = pkl.load(handle)
            print("cache loaded")
        except:
            cache = None
        df = ori_df.sample(frac=sample_frac)
        for attribute in attributes:
            if attribute == "dataset" or attribute == "class":
                cmap = matplotlib.cm.get_cmap("tab10")
            elif attribute == "K=20":
                cmap = matplotlib.cm.get_cmap("tab20")
            else:
                cmap = matplotlib.cm.get_cmap("jet")

            fname_output = "CL_tsne_%s_color_%s" % ("all", attribute)
            img_list = []
            r_list = []
            c_list = []
            for _, row in df.iterrows():
                fname_img = os.path.join(
                    root_path,
                    row[col_dataset],
                    # 'input_neuron',
                    "swc_soma0_screenshot",
                    os.path.basename(row[col_file]) + "_%s.png" % view,
                )
                # sub_img.append((255-img[350:650,350:650,:],None))
                if attribute == "dataset":
                    color = [
                        1 - _ for _ in cmap(dataset_dict[row[attribute]])[:3][::-1]
                    ]
                elif attribute in [
                    "class",
                    "K=20",
                    "height",
                    "n_nodes",
                    "avg_length",
                    "total_length",
                    "avg_contraction",
                    "std_contraction",
                ]:
                    color = [1 - _ for _ in cmap(row[attribute])[:3][::-1]]
                else:
                    ## random color
                    color = np.random.random((1, 1, 3))
                    color /= np.max(color)  # make it dark
                # color -= np.min(color) # make it bright
                # color /= np.max(color) # make it dark
                img_list.append((fname_img, color))
                r_list.append(row[col_r] * scale_coord)
                c_list.append(row[col_c] * scale_coord)
            plot_tsne_img(
                fname_output + "_%s.png" % view,
                img_list,
                r_list,
                c_list,
                cache,
                task_name=f"{attribute}_{attribute}_{view} ({k}/{len(views)*len(attributes)})",
            )
            k += 1
        del cache


# script_plot_tsne2()


def script_plot_cluster():
    root_path = "neuron_morpho_dendrite"

    fname_csv = os.path.join(root_path, "results/5_feats.csv")
    fname_output = fname_csv + "_classes"
    max_sample_per_cluster = 15
    min_cluster_size = 20

    df = pd.read_csv(fname_csv)
    col_cluster = "class_name"
    col_file = "file_name"
    col_dataset = "dataset"

    # df = df[df['class_name']=='VPM']
    k_list = df[col_cluster].unique()
    k_list.sort()
    np.random.seed(0)
    colors = [np.random.random((1, 1, 3)) for _ in range(3)]
    cluster_img_view = {"xy": [], "yz": [], "xz": []}
    for k in k_list:
        print(k)
        sub_df = df[df[col_cluster] == k]
        if len(sub_df) < min_cluster_size:
            continue
        sub_df = sub_df.sample(n=max_sample_per_cluster)
        sub_img_view = {key: [] for key in cluster_img_view}
        for _, row in sub_df.iterrows():
            for ix, view in enumerate(cluster_img_view):
                fname_img = os.path.join(
                    root_path,
                    row[col_dataset],
                    "swc_soma0_screenshot",
                    os.path.basename(row[col_file]) + "_%s.png" % view,
                )
                # sub_img.append((255-img[350:650,350:650,:],None))
                ## random color
                # color = np.random.random((1,1,3))
                color = colors[ix]
                color /= np.max(color)
                ## color by tsne coordinate
                # color = np.ones((1,1,3))
                # color[...,1] = max(0,min(1,(row['x']+64)/128))
                # color[...,2] = max(0,min(1,(row['y']+64)/128))
                sub_img_view[view].append((fname_img, color))
        for view in cluster_img_view:
            cluster_img_view[view].append(sub_img_view[view])
    for view in cluster_img_view:
        plot_clusters_img(fname_output + "_%s.png" % view, cluster_img_view[view])


# script_plot_cluster()


def cache_img():
    from glob import glob
    import pickle as pkl
    import cv2

    root_path = "neuron_morpho_dendrite"
    fname_csv = os.path.join(root_path, "results/5_feats.csv")
    df = pd.read_csv(fname_csv)
    col_dataset = "dataset"
    col_file = "file_name"
    for view in ["xy", "xz", "yz"]:
        cache = {}
        for _, row in tqdm(df.iterrows(), total=len(df)):
            fname_img = os.path.join(
                root_path,
                row[col_dataset],
                # 'input_neuron',
                "swc_soma0_screenshot",
                os.path.basename(row[col_file]) + "_%s.png" % view,
            )
            cache[fname_img] = cv2.imread(fname_img)
        with open("swc_screenshot_%s.pkl" % view, "wb") as handle:
            pkl.dump(cache, handle)


# cache_img()
