from models.components.gnns.gat import GatMeanPool
from models.components.language.lstm import Lstm
from models.base_captioner import BaseCaptioner

import torch

class SpatialSemanticGat(BaseCaptioner):
    def __init__(self, 
                 embedding_size: int, 
                 hidden_size: int, 
                 vocab_size: int, 
                 num_layers: int) -> None:
        super(SpatialSemanticGat, self).__init__(embedding_size, 
                                                      hidden_size, 
                                                      vocab_size, 
                                                      num_layers)
        self.encoder_spatial = GatMeanPool(embedding_size, embedding_size)
        self.encoder_semantic = GatMeanPool(embedding_size, embedding_size)
        self.decoder = Lstm(embedding_size, hidden_size, vocab_size, num_layers)
    
    
    def forward(self, graphs, captions):   
        spatial_graphs = graphs[0]
        semantic_graphs = graphs[1]
        
        spatial_features = self.encoder_spatial(spatial_graphs)
        semantic_features = self.encoder_semantic(semantic_graphs)

        features = (spatial_features + semantic_features) / 2

        outputs = self.decoder(features, captions)
        return outputs
    

    def caption_image(self, input_features, vocab, max_length=50):
        with torch.no_grad():            
            spatial_features = self.encoder_spatial(input_features[0])
            semantic_features = self.encoder_semantic(input_features[1])

            x = (spatial_features + semantic_features) / 2
            states = None

            result = []
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(0)
                result.append(predicted.item())

                x = self.decoder.embedding(predicted).unsqueeze(0)
                if vocab.itos[f"{predicted.item()}"] == "<EOS>":
                    break
            return [vocab.itos[f"{idx}"] for idx in result]


