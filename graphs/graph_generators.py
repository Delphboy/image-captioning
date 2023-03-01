import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.autograd import Variable
from torch_geometric.data import Batch, Data
from torchvision.models import ResNet50_Weights, ResNet101_Weights, resnet50
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)

from constants import Constants as const


class SpatialGraphGenerator():
    def __init__(self):        
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    
    def _get_iou(self, box1, box2):
        """
        Implement the intersection over union (IoU) between box1 and box2

        Arguments:
            box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
            box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        """
        # ymin, xmin, ymax, xmax = box

        y11, x11, y21, x21 = box1
        y12, x12, y22, x22 = box2

        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area
        # compute the IoU
        iou = inter_area / union_area
        return iou
    

    def _convert_to_pyg(self, nodes, adj_mat, image_prediction):
        node_features = torch.stack([image_prediction['features'][node] for node in nodes])
        node_features = node_features.squeeze(1)
        node_features = node_features.to(const.DEVICE)

        froms = []
        tos = []
        edge_attrs = []
        for i in range(len(adj_mat)):
            from_edge = adj_mat[i]
            for j in range(len(from_edge)):
                froms.append(i)
                tos.append(j)
                edge_attrs.append([adj_mat[i][j]])

        edges = torch.stack([torch.tensor(froms), torch.tensor(tos)])
        edge_attrs = torch.tensor(edge_attrs).to(const.DEVICE)

        return Data(x=node_features, edge_index=edges, edge_attr=edge_attrs).to(const.DEVICE)

    # TODO: Can we speed this up?
    def _generate_spatial_graph(self, image_prediction):
        nodes = [i for i in range(image_prediction['boxes'].shape[0])]
        edges = np.zeros((len(nodes),len(nodes)))
        
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    bbox_a = image_prediction['boxes'][i]
                    bbox_b = image_prediction['boxes'][j]

                    x_a1 = int(bbox_a[0])
                    y_a1 = int(bbox_a[1])
                    x_a2 = int(bbox_a[2])
                    y_a2 = int(bbox_a[3])

                    x_b1 = int(bbox_b[0])
                    y_b1 = int(bbox_b[1])
                    x_b2 = int(bbox_b[2])
                    y_b2 = int(bbox_b[3])

                    # Inside
                    if (x_a1 > x_b1 and x_a2 < x_b2) and (y_a1 > y_b1 and y_a2 < y_b2):
                        edges[i,j] = 1
                        continue

                    # Cover
                    if (x_b1 > x_a1 and x_b2 < x_a2) and (y_b1 > y_a1 and y_b2 < y_a2):
                        edges[i,j] = 2
                        continue

                    # Overlap
                    iou = self._get_iou(bbox_a, bbox_b)
                    if iou >= 0.5:
                        edges[i,j] = 3
                        continue
                    else:
                        centroid_a = np.array([int(x_a1 + abs(x_a1 - x_a2) / 2),int(y_a1 + abs(y_a1 - y_a2) / 2)])
                        centroid_b = np.array([int(x_b1 + abs(x_b1 - x_b2) / 2),int(y_b1 + abs(y_b1 - y_b2) / 2)])

                        vecAB = centroid_b - centroid_a
                        #deg = np.angle([vec[0] + vec[1]j], deg=True)

                        hoz = np.array([1, 0])

                        inner = np.inner(vecAB, hoz)
                        norms = np.linalg.norm(vecAB) * np.linalg.norm(hoz)

                        cos = inner / norms
                        rad = np.arccos(np.clip(cos, -1.0, 1.0))
                        deg = np.rad2deg(rad)            

                        edges[i,j] = np.ceil(deg/45) + 3
                        continue
        
        return self._convert_to_pyg(nodes, edges, image_prediction)


    def generate_spatial_graph_for_batch(self, image_predictions):
        graphs = []
        for image_prediction in image_predictions:
            graphs.append(self._generate_spatial_graph(image_prediction))

        batch = Batch.from_data_list(graphs)
        return batch
        
    