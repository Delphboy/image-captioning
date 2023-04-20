import torch
import torch.nn as nn

from graphs.spatial_graph_generator import *
from models.components.gnns.gcn import Gcn
from models.components.language.lstm import Lstm
from models.components.vision.object_detectors import \
    FasterRcnnResNet101BoundingBoxes


class CaptionWithSpatialGraph(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithSpatialGraph, self).__init__()
        self.cnn = FasterRcnnResNet101BoundingBoxes(embedding_size=embed_size)
        self.gcn = Gcn(embed_size, embed_size)
        self.decoder = Lstm(embed_size, hidden_size, vocab_size, num_layers)
        self.pool = nn.MaxPool1d(1)
        self.spaital_graph_generator = SpatialGraphGenerator()
    
    
    def forward(self, images, captions, spatial_graphs):    
        # Apply GCN
        features = self.gcn(spatial_graphs)            
        
        # Apply LSTM
        outputs = self.decoder(features, captions)
        return outputs
    

    def caption_image(self, images, vocab, max_length=50):
        with torch.no_grad():
            spatial_graphs = self.spaital_graph_generator.generate_spatial_graph_for_batch(images)
            return self.caption_image_precomputed(self, spatial_graphs, vocab, max_length)
            

    def caption_image_precomputed(self, spatial_graphs, vocab, max_length=50):
        with torch.no_grad():
            result = []
            x = self.gcn(spatial_graphs)
            states = None
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(0)
                result.append(predicted.item())

                x = self.decoder.embedding(predicted).unsqueeze(0)
                if vocab.itos[f"{predicted.item()}"] == "<EOS>":
                    break
            return [vocab.itos[f"{idx}"] for idx in result]

