import torch
import torch.nn as nn

from models.components.vision.object_detectors import FasterRcnnResNet101BoundingBoxes
from models.components.language.lstm import Lstm

from graphs.spatial_graph_generator import SpatialGraphGenerator
from models.components.gnns.gcn import Gcn


class CaptionWithSpatialGraph(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithSpatialGraph, self).__init__()
        self.cnn = FasterRcnnResNet101BoundingBoxes(embedding_size=256)
        self.gcn = Gcn(embed_size, embed_size)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)
        self.pool = nn.MaxPool1d(1)
        self.spaital_graph_generator = SpatialGraphGenerator()
    
    
    def forward(self, images, captions, spatial_graphs):    
        # Apply GCN
        features = self.gcn(spatial_graphs)            
        
        # Apply LSTM
        outputs = self.lstm(features, captions)
        return outputs
    

    def caption_image(self, images, vocabulary, max_length=50):
        with torch.no_grad():
            spatial_graphs = self.spaital_graph_generator.generate_spatial_graph_for_batch(images)
            features = self.gcn(spatial_graphs.x, spatial_graphs.edge_index, spatial_graphs.batch)
            result_caption = self.lstm.sample(features)[0]
        return [vocabulary.itos[str(idx.item())] for idx in result_caption]
    

    def caption_image_precomputed(self, spatial_graphs, vocabulary, max_length=50):
        with torch.no_grad():
            features = self.gcn(spatial_graphs)
            result_caption = self.lstm.sample(features)
        
        return [vocabulary.itos[str(idx.item())] for idx in result_caption[0]]

