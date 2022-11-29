from clustering import Kmeans
# from fast_pytorch_kmeans import KMeans

import argparse
import os
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tree_dataset import NeuronTreeDataset, get_collate_fn, LABEL_DICT
from treelstm import TreeLSTM
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

all_wo_others = {'VPM': 0, 'Isocortex_layer23': 1, 'Isocortex_layer4': 2, 
                 'PRE': 3, 'SUB': 4, 'CP': 5, 'VPL': 6, 'Isocortex_layer6': 7, 
                 'MG': 8, 'Isocortex_layer5': 9}
label_idx_to_class_names = {v:k for k,v in all_wo_others.items()}
NeuronBatch = namedtuple('NeuronBatch', ['graph', 'feats', 'label', 'offset','fname', 'idxs'])

def collate_fn(batch):
        trees, offsets, labels, fnames = [], [0], [], []
        idxs = []
        cnt = 0
        for b in batch:
            trees.append(b[0])
            cnt += b[1]
            offsets.append(cnt)
            labels.append(b[2])
            fnames.append(b[3])
            idxs.append(b[4])
        offsets.pop(-1)
        batch_trees = dgl.batch(trees)
        return NeuronBatch(
                    graph=batch_trees.to('cuda'),
                    feats=batch_trees.ndata['feats'].float().to('cuda'),
                    label=torch.as_tensor(labels).to('cuda'),
                    offset=torch.IntTensor(offsets).to('cuda'),
                    fname=fnames,
                    idxs=torch.IntTensor(idxs).long().to('cuda'))

class DebugNeuronTreeDataset(NeuronTreeDataset):
    def __init__(self,
                 phase='full',
                 input_features=[2,3,4],
                 topology_transformations=None,
                 attribute_transformations=None,
                 dataset='all_wo_others',
                 data_dir = None,
                 label_dict=all_wo_others):
        assert phase in ['train','test', 'full']
        super(DebugNeuronTreeDataset, self).__init__(
            phase=phase,
            input_features=input_features,
            topology_transformations=topology_transformations,
            attribute_transformations=attribute_transformations,
            dataset=dataset,
            data_dir=data_dir,
            label_dict=label_dict)

    def __getitem__(self, idx):
        return self.neuron_trees[idx], self.tree_lens[idx], self.targets[idx], self.file_list[idx], idx

def extract2(model, loader, batch0):
    # loader = DataLoader(
    #                     dataset=dataset,
    #                     batch_size=512,
    #                     collate_fn=collate_fn,
    #                     shuffle=True)
    features = torch.zeros((len(loader.dataset),128)).cuda()
    net = model.encoder_q
    
    net.eval()
    with torch.no_grad():
        _ = net(batch0)
        for batch in tqdm(loader, desc='Feature extracting'):
            feature = net(batch)
            features[batch.idxs,:]=feature
        print(batch.idxs[-1])
    features = nn.functional.normalize(features,dim=1)
    return features

        
def extract(model, loader, batch0):
    features = torch.zeros((len(loader.dataset),128)).cuda()
    net = model.encoder_q
    net.eval()
    with torch.no_grad():
        _ = net(batch0)
        for batch in tqdm(loader, desc='Feature extracting'):
            feature = net(batch)
            features[batch.idxs,:]=feature
        print(batch.idxs[-1])
    features = nn.functional.normalize(features,dim=1)
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Supervised Training')
    # Basic
    parser.add_argument('--dataset', type=str, 
                        default='all_wo_others')
    # Model 
    parser.add_argument('--model', type=str, default='double')
    parser.add_argument('--child_mode', type=str, default='sum', help='[sum, average]')
    parser.add_argument('--input_features', nargs="+", type=int, default=[2,3,4,12,13], help='selected columns')
    parser.add_argument('--h_size', type=int, default=128, help='memory size for lstm')
    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training [default: 128]')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    # MoCo specific configs:
    parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco-k', default=1024, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')
    parser.add_argument('--symmetric', action='store_true', default=False,
                        help='use a symmetric loss function that backprops to both views')
    # others
    parser.add_argument('--eval_part', type=str, default='full', help='[full | backbone]')
    parser.add_argument('--use_translation_feats', action='store_true', default=False,
                        help='use 24-d translation feats')
    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
    # work_dir/fullset/base_wt_others_unbalanced/epoch_40.pth

    args = parser.parse_args()
    args.bn = True
    args.projector_bn = True
    args.use_translation_feats = True
    # args.input_features = [2,3,4]
    args.pretrained = 'work_dir/arch_ablation/full_double_29_bn_proj_bn_all/epoch_90.pth'
    # args.pretrained = 'work_dir/arch_ablation/full_double_5_bn_proj_bn_all-drop_0_0_0.005/epoch_65.pth'
    # args.pretrained = 'work_dir/topo_ablation/double_dummy_coords_normal/epoch_75.pth'
    if args.use_translation_feats:
        args.input_features+=[i for i in range(20,44)]
    # hyper parameters
    device = torch.device('cuda')
    # create the model
    print(json.dumps(vars(args), indent=4, sort_keys=True))  # print args
    print("=> creating models ...")
    model = MoCoTreeLSTM(
        args, # for TreeLSTM
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        symmetric=args.symmetric).to(device)
    print(model)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrained)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.pretrained, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrained, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    cudnn.benchmark = True
    if args.eval_part == 'backbone':
        net = model.backbone_q
    elif args.eval_part == 'full':
        net = model.encoder_q
    else:
        raise ValueError(f'eval part must be in [backbone|full], got {args.eval_part}')

    dataset = DebugNeuronTreeDataset(
                        phase='full',
                        input_features=args.input_features,
                        dataset=args.dataset,
                        label_dict=LABEL_DICT[args.dataset])  
    dummy_loader1 = DataLoader(
                        dataset=dataset,
                        batch_size=1,
                        collate_fn=collate_fn,
                        shuffle=True)
    dummy_loader2 = DataLoader(
                        dataset=dataset,
                        batch_size=2,
                        collate_fn=collate_fn,
                        shuffle=True)
    loader1 = DataLoader(
                        dataset=dataset,
                        batch_size=512,
                        collate_fn=collate_fn,
                        shuffle=True)
    loader2 = DataLoader(
                        dataset=dataset,
                        batch_size=256,
                        collate_fn=collate_fn,
                        shuffle=True)
    
    for batch in dummy_loader1:
        batch0 = batch
        break
    for batch in dummy_loader2:
        batch1 = batch
        break

    feats1 = extract(model, loader1, batch0)
    feats2 = extract2(model, loader2, batch1)
    print(torch.norm(feats1-feats2))
    import pdb; pdb.set_trace()
    print(111)

