from models.components.vision.encoders import InceptionV3, Resnet
from models.components.language.lstm import Lstm
from models.Captioner import BaseCaptioner


class CaptionWithInceptionV3AndLstm(BaseCaptioner):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithInceptionV3AndLstm, self).__init__(embed_size, 
                                                         hidden_size, 
                                                         vocab_size, 
                                                         num_layers)
        self.encoder = InceptionV3(embed_size)
        self.decoder = Lstm(embed_size, hidden_size, vocab_size, num_layers)



class CaptionWithResnet152AndLstm(BaseCaptioner):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithResnet152AndLstm, self).__init__(embed_size, 
                                                         hidden_size, 
                                                         vocab_size, 
                                                         num_layers)
        self.encoder = Resnet(embed_size, 152)
        self.decoder = Lstm(embed_size, hidden_size, vocab_size, num_layers)



class CaptionWithResnet101AndLstm(BaseCaptioner):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithResnet101AndLstm, self).__init__(embed_size, 
                                                         hidden_size, 
                                                         vocab_size, 
                                                         num_layers)
        self.encoder = Resnet(embed_size, 101)
        self.decoder = Lstm(embed_size, hidden_size, vocab_size, num_layers)




class CaptionWithResnet18AndLstm(BaseCaptioner):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithResnet18AndLstm, self).__init__(embed_size, 
                                                         hidden_size, 
                                                         vocab_size, 
                                                         num_layers)
        self.max_length=20
        self.encoder = Resnet(embed_size, 18)
        self.decoder = Lstm(embedding_size=embed_size, 
                         hidden_size=hidden_size, 
                         vocab_size=vocab_size, 
                         num_layers=num_layers)


