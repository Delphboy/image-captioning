from models.components.gnns.gat import GatMeanPool
from models.components.language.lstm import Lstm
from models.base_captioner import BaseCaptioner


class SpatialGat(BaseCaptioner):
    def __init__(self, 
                 embedding_size: int, 
                 hidden_size: int, 
                 vocab_size: int, 
                 num_layers: int) -> None:
        super(SpatialGat, self).__init__(embedding_size, 
                                         hidden_size, 
                                         vocab_size, 
                                         num_layers)
        self.encoder = GatMeanPool(embedding_size, embedding_size)
        self.decoder = Lstm(embedding_size, hidden_size, vocab_size, num_layers)
    
    
    def forward(self, graphs, captions):   
        return super().forward(graphs[0], captions) 
    

    def caption_image(self, graphs, vocab, max_lenth=50):
        return super().caption_image(graphs[0], vocab, max_lenth)


class SemanticGat(BaseCaptioner):
    def __init__(self, 
                 embedding_size: int, 
                 hidden_size: int, 
                 vocab_size: int, 
                 num_layers: int) -> None:
        super(SemanticGat, self).__init__(embedding_size, 
                                         hidden_size, 
                                         vocab_size, 
                                         num_layers)
        self.encoder = GatMeanPool(embedding_size, embedding_size)
        self.decoder = Lstm(embedding_size, hidden_size, vocab_size, num_layers)
    
    
    def forward(self, graphs, captions):   
        return super().forward(graphs[1], captions) 
    

    def caption_image(self, graphs, vocab, max_lenth=50):
        return super().caption_image(graphs[1], vocab, max_lenth)
