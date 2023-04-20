import json
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from constants import Constants as const
from factories.data_factory import get_data
from models.components.vision.object_detectors import FasterRcnnResNet101BoundingBoxes
from torchvision.ops.boxes import box_iou
from utils.helper_functions import *


class SpatialGraphGenerator():
    def __init__(self, 
                 embedding_size=2048):
        self.detector = FasterRcnnResNet101BoundingBoxes(embedding_size).to(const.DEVICE)
        self.detector.eval()
        self.relationship_weights = self._load_relationship_weights()
   

    def _convert_to_pyg(self, nodes, adj_mat, image_prediction, include_relationship_weights=False):
        node_features = torch.stack([image_prediction['features'][node] for node in nodes])
        node_features = node_features.squeeze(1)
        node_features = node_features.to(const.DEVICE)


        froms = []
        tos = []
        edge_attrs = []
        for i in range(len(adj_mat)):
            from_edge = adj_mat[i]
            for j in range(len(from_edge)):
                if i == j: continue
                froms.append(i)
                tos.append(j)
                if include_relationship_weights:
                    relationship_weight = 0.0
                    if len(image_prediction['labels']) > 0:
                        from_obj = image_prediction['labels'][i]
                        to_obj = image_prediction['labels'][j]
                        key = f"({from_obj}, {to_obj})"
                        if key in self.relationship_weights.keys():
                            relationship_weight = self.relationship_weights[key]
                    edge_attrs.append([adj_mat[i][j], relationship_weight])
                else:
                    edge_attrs.append([adj_mat[i][j]])

        if len(froms) == 0 and len(tos) == 0:
            froms.append(0)
            tos.append(0)
            edge_attrs.append([0])


        edges = torch.stack([torch.tensor(froms), torch.tensor(tos)])
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float).to(const.DEVICE)

        return Data(x=node_features, edge_index=edges, edge_attr=edge_attrs).to(const.DEVICE)


    def _load_relationship_weights(self):
        with open('datasets/relationship_weights.json') as f:
            relationship_weights = json.load(f)
        return relationship_weights


    # TODO: Can we speed this up?
    def _generate_spatial_graph(self, image_prediction, include_relationship_weights=False):
        nodes = [a for a in range(image_prediction['boxes'].shape[0])]
        edges = np.zeros((len(nodes),len(nodes)))
        
        for a in range(len(nodes)):           
            for b in range(len(nodes)):
                if a == b: continue

                bbox_a = image_prediction['boxes'][a]
                bbox_a_x1, bbox_a_y1, bbox_a_x2, bbox_a_y2 = bbox_a

                bbox_b = image_prediction['boxes'][b]
                bbox_b_x1, bbox_b_y1, bbox_b_x2, bbox_b_y2 = bbox_b

                # Check if bbox_a is inside of bbox_b
                if bbox_a_x1 > bbox_b_x1 and bbox_a_y1 > bbox_b_y1 and bbox_a_x2 < bbox_b_x2 and bbox_a_y2 < bbox_b_y2:
                    edges[a][b] = 1

                # Check if bbox_a is outside of bbox_b
                elif bbox_a_x1 < bbox_b_x1 and bbox_a_y1 < bbox_b_y1 and bbox_a_x2 > bbox_b_x2 and bbox_a_y2 > bbox_b_y2:
                    edges[a][b] = 2

                # Check if bbox_a and bbox_b have an IoU of >= 0.5
                elif box_iou(bbox_a.unsqueeze(0), bbox_b.unsqueeze(0)) >= 0.5:
                    edges[a][b] = 3

                else:
                    centroid_a = torch.tensor([bbox_a_x1 + abs(bbox_a_x1 - bbox_a_x2) / 2, bbox_a_y1 + abs(bbox_a_y1 - bbox_a_x2) / 2])
                    centroid_b = torch.tensor([bbox_b_x1 + abs(bbox_b_x1 - bbox_b_x2) / 2, bbox_b_y1 + abs(bbox_b_y1 - bbox_b_y2) / 2])

                    vecAB = centroid_b - centroid_a
                    hoz = torch.tensor([1, 0], dtype=torch.float)

                    inner = torch.inner(vecAB, hoz)
                    norms = torch.linalg.norm(vecAB) * torch.linalg.norm(hoz)

                    cos = inner / norms
                    rad = torch.acos(torch.clamp(cos, -1.0, 1.0))
                    deg = torch.rad2deg(rad)

                    edges[a,b] = torch.ceil(deg/45) + 3

        if len(nodes) == 1:
            edges[0][0] = 0

        return self._convert_to_pyg(nodes, 
                                    edges, 
                                    image_prediction, 
                                    include_relationship_weights)


    def generate_spatial_graph_for_batch(self, image_batch):
        graphs = []
        image_predictions = self.detector(image_batch)
        for image_prediction in image_predictions:
            graphs.append(self._generate_spatial_graph(image_prediction))

        batch = Batch.from_data_list(graphs)
        return batch
        

def generate_spatial_graphs_on_flickr8k(split="train",
                                        include_relationship_weights=False):
    const.DATASET = 'flickr8k'
    const.ROOT = "/import/gameai-01/eey362/datasets/flickr8k/images"
    const.ANNOTATIONS = "/homes/hps01/image-captioning/datasets/splits/dataset_flickr8k.json"
    const.TALK_FILE = "/homes/hps01/image-captioning/datasets/talk_files/flickrtalk.json"
    const.BATCH_SIZE = 1
    const.NUM_WORKERS = 0
    const.SHUFFLE = False


    train_loader, val_loader, test_loader, _, _, _, _ = get_data('flickr8k')

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    assert split in loaders.keys(), f"Split must be one of {loaders.keys()}"
    loader = loaders[split]

    generator = SpatialGraphGenerator()

    if include_relationship_weights:
        save_name = f"precomputed_graphs/flickr8k_spatial_{split}_with_weights.pt"
    else:
        save_name = f"precomputed_graphs/flickr8k_spatial_{split}.pt"

    results = {}


    idx = 0
    for idx, data in enumerate(loader):
        image_id = loader.dataset.ids[idx]
        print(f"Processing {idx+1}th image, image_id: {image_id}")
        
        images = data[0].to(const.DEVICE)
        
        image_prediction = generator.detector(images)[0]
        spatial_graph = generator._generate_spatial_graph(image_prediction, include_relationship_weights)
        
        results[image_id] = spatial_graph
        idx += 1

    torch.save(results, f'saves/{save_name}')


def generate_spatial_graphs_on_coco_karpathy(split="val",
                                             include_relationship_weights=False):
    const.DATASET = 'coco_karpathy'
    const.ROOT = "/import/gameai-01/eey362/datasets/coco/images"
    const.ANNOTATIONS = "/homes/hps01/image-captioning/datasets/splits/dataset_coco.json"
    const.TALK_FILE = "/homes/hps01/image-captioning/datasets/talk_files/cocotalk.json"
    const.BATCH_SIZE = 1
    const.SHUFFLE = False

    train_loader, val_loader, test_loader, _, _, _, _ = get_data('coco_karpathy')

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    assert split in loaders.keys(), f"Split must be one of {loaders.keys()}"
    loader = loaders[split]

    generator = SpatialGraphGenerator()

    if include_relationship_weights:
        save_name = f"coco_karpathy_spatial_{split}_with_weights.pt"
    else:
        save_name = f"coco_karpathy_spatial_{split}.pt"

    results = {}

    for idx, _ in enumerate(loader):
        image_id = loader.dataset.ids[idx]
        print(f"Processing {idx+1}th image, image_id: {image_id}")

        attribs = read_bottom_up_top_down_features(image_id)
        spatial_graph = generator._generate_spatial_graph(attribs, include_relationship_weights)

        results[image_id] = spatial_graph
    
    save_loc = os.path.join(const.ROOT, '..', "precomputed_graphs", save_name)
    torch.save(results, save_loc)
