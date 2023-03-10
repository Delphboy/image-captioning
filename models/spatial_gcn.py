import torch.nn as nn

from models.components.vision.object_detectors import FasterRcnnResNet101BoundingBoxes
from models.components.language.lstm import Lstm

from graphs.spatial_graph_generator import SpatialGraphGenerator
from models.components.gnns.gcn import Gcn


class CaptionWithSpatialGraph(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithSpatialGraph, self).__init__()
        self.cnn = FasterRcnnResNet101BoundingBoxes(embed_size)
        self.gcn = Gcn(embed_size, embed_size)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)

        self.spaital_graph_generator = SpatialGraphGenerator()
    
    
    def forward(self, images, captions, lengths, graphs):
        # Get objects
        # image_predictions = self.cnn(images)
        
        # Generate spatial graph
        spatial_graphs = graphs#self.spaital_graph_generator.generate_spatial_graph_for_batch(image_predictions)        
    
        # Apply GCN
        features = self.gcn(spatial_graphs.x, spatial_graphs.edge_index, spatial_graphs.batch)            
        
        # Apply LSTM
        outputs = self.lstm(features, captions, lengths)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        image_predictions = self.cnn(image)
        spatial_graphs = self.spaital_graph_generator.generate_spatial_graph_for_batch(image_predictions)
        features = self.gcn(spatial_graphs.x, spatial_graphs.edge_index, spatial_graphs.batch)
        result_caption = self.lstm.sample(features)[0]
        
        return [vocabulary.itos[str(idx.item())] for idx in result_caption]
    

    def caption_image_precomputed(self, spatial_graphs, vocabulary, max_length=50):
        features = self.gcn(spatial_graphs.x, spatial_graphs.edge_index, spatial_graphs.batch)
        result_caption = self.lstm.sample(features)[0]
        
        return [vocabulary.itos[str(idx.item())] for idx in result_caption]