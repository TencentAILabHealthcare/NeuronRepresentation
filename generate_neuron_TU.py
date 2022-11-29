import pickle as pkl
import numpy as np
import os

for dataset in ['seu_6_classes', 'JM', 'ACT']:
    for split in ['train', 'test']:
        dataset_name = f'{dataset}_{split}'
        print(dataset_name)
        saved_cache = f'datasets/neuron_morpho/processed_datasets_bk2/{dataset_name}.pkl'
        with open(saved_cache, 'rb') as f:
            saved_cache = pkl.load(f)
        lines, i2ps, targets = saved_cache['lines'], saved_cache['i2ps'], saved_cache['targets'] 
        file_list = saved_cache['file_list']
        os.makedirs(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw', exist_ok=True)        
        adj = open(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw/{dataset_name}_A.txt', 'w')
        graph_ids = open(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw/{dataset_name}_graph_indicator.txt', 'w')
        graph_labels = open(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw/{dataset_name}_graph_labels.txt', 'w')
        node_attributes = open(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw/{dataset_name}_node_attributes.txt', 'w')

        input_features = [2,3,4,12,13]
        input_features += [i for i in range(20,44)]

        offset, graph_id = 0, 0
        for neuron_graph, label in zip(lines, targets):
            graph_id += 1
            graph_labels.write(f'{label}\n')
            for ix, node_attr in enumerate(neuron_graph):
                # node attribute
                attrs = [str(float(x)) for x in node_attr[input_features]]
                attr_msg = ','.join(attrs) + '\n'
                node_attributes.write(attr_msg)
                
                # graph_id
                graph_ids.write(f'{graph_id}\n')
                
                # graph labels
                # adj
                if ix != 0:
                    adj.writelines([f'{int(node_attr[0])+offset}, {int(node_attr[6]+offset)}\n',
                                    f'{int(node_attr[6])+offset}, {int(node_attr[0]+offset)}\n'])        
            offset += len(neuron_graph)
            
lines, targets = [], []
for dataset in ['seu_6_classes', 'JM', 'ACT']:
    dataset_name = f'{dataset}_train'
    print(dataset_name)
    saved_cache = f'datasets/neuron_morpho/processed_datasets_bk2/{dataset_name}.pkl'
    with open(saved_cache, 'rb') as f:
        saved_cache = pkl.load(f)
    _lines, _targets = saved_cache['lines'], saved_cache['targets'] 
    lines.extend(_lines)
    targets.extend(_targets)
    # print(type(lines), type(targets))
    # exit()
dataset_name = 'train_3'
os.makedirs(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw', exist_ok=True)        
adj = open(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw/{dataset_name}_A.txt', 'w')
graph_ids = open(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw/{dataset_name}_graph_indicator.txt', 'w')
graph_labels = open(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw/{dataset_name}_graph_labels.txt', 'w')
node_attributes = open(f'GraphCL/unsupervised_TU/data/neuron/{dataset_name}/raw/{dataset_name}_node_attributes.txt', 'w')

input_features = [2,3,4,12,13]
input_features += [i for i in range(20,44)]

offset, graph_id = 0, 0
for neuron_graph, label in zip(lines, targets):
    graph_id += 1
    graph_labels.write(f'{label}\n')
    for ix, node_attr in enumerate(neuron_graph):
        # node attribute
        attrs = [str(float(x)) for x in node_attr[input_features]]
        attr_msg = ','.join(attrs) + '\n'
        node_attributes.write(attr_msg)
        
        # graph_id
        graph_ids.write(f'{graph_id}\n')
        
        # graph labels
        # adj
        if ix != 0:
            adj.writelines([f'{int(node_attr[0])+offset}, {int(node_attr[6]+offset)}\n',
                            f'{int(node_attr[6])+offset}, {int(node_attr[0]+offset)}\n'])        
    offset += len(neuron_graph)
                
        
        