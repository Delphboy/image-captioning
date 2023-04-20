import os

import numpy as np
import torch

from factories.data_factory import get_data
from torch_geometric.data import Data
from utils.helper_functions import read_coco_karpathy_attributes
from constants import Constants as const

directory = "/import/gameai-01/eey362/datasets/coco/data/"
dictionary_file = "coco_pred_sg_rela.npy"
attribute_folder = "cocobu_att"
bbox_folder = "cocobu_box"
image_feature_folder = "cocobu_fc"
semantic_graph_folder = "coco_img_sg"


def read_coco_karpathy_dictionary() -> dict:
    dictionary_file_loc = os.path.join(directory, dictionary_file)
    dictionary = np.load(dictionary_file_loc, allow_pickle=True, encoding='bytes').item()
    return dictionary


def read_semantic_graph(image_id):
    semantic_graph_file_loc = os.path.join(directory, semantic_graph_folder, f'{image_id}.npy')
    semantic_graph = np.load(semantic_graph_file_loc, allow_pickle=True, encoding='bytes').item()
    return semantic_graph


def generate_semantic_graphs_on_coco_karpathy(split: str = 'train'):
    const.DATASET = 'coco_karpathy'
    const.ROOT = "/import/gameai-01/eey362/datasets/coco/images"
    const.ANNOTATIONS = "/homes/hps01/image-captioning/datasets/splits/dataset_coco.json"
    const.TALK_FILE = "/homes/hps01/image-captioning/datasets/talk_files/cocotalk.json"
    const.BATCH_SIZE = 1
    const.SHUFFLE = False

    assert split in ['train', 'val', 'test'], "Invalid split"
    
    train_loader, val_loader, test_loader, _, _, _, _ = get_data('coco_karpathy')
    
    semantic_graphs = {}

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    loader = loaders[split]

    for idx, _ in enumerate(loader):
        image_id = loader.dataset.ids[idx]
        
        print(f"Processing {idx+1}th image, image_id: {image_id}")
        
        attribs = read_coco_karpathy_attributes(directory, attribute_folder, image_id)
        edges_np = read_semantic_graph(image_id)[b'rela_matrix']
        
        x = torch.tensor(attribs)
        _t = [e[:2] for e in edges_np]
        edges = torch.tensor(_t).T
        edge_attr = torch.tensor([edge[2] for edge in edges_np])

        graph = Data(x=x, edge_index=edges, edge_attr=edge_attr)
        semantic_graphs[image_id] = graph

    torch.save(semantic_graphs, os.path.join(directory, f"/import/gameai-01/eey362/datasets/coco/precomputed_graphs/coco_karpathy_semantic_{split}.pt"))

