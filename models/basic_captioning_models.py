from typing import Optional
import torch
import torch.nn as nn

from models.components.vision.encoders import InceptionV3, Resnet
from models.components.vision.object_detectors import FasterRcnnResNet101BoundingBoxes
from models.components.language.lstm import Lstm

from graphs.spatial_graph_generator import SpatialGraphGenerator
from models.components.gnns.gcn import Gcn


class CaptionWithInceptionV3AndLstm(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithInceptionV3AndLstm, self).__init__()
        self.cnn = InceptionV3(embed_size)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)


    def forward(self, images, captions, lengths):
        features = self.cnn(images)
        outputs = self.lstm(features, captions, lengths)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        features = self.cnn(image).unsqueeze(0)
        result_caption = self.lstm.sample(features)[0]
        
        return [vocabulary.itos[idx.item()] for idx in result_caption]


class CaptionWithResnet152AndLstm(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithResnet152AndLstm, self).__init__()
        self.cnn = Resnet(embed_size, 152)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.cnn(images)
        outputs = self.lstm(features, captions, lengths)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        features = self.cnn(image).unsqueeze(0)
        result_caption = self.lstm.sample(features)[0]
        
        return [vocabulary.itos[idx.item()] for idx in result_caption]


class CaptionWithResnet101AndLstm(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithResnet101AndLstm, self).__init__()
        self.cnn = Resnet(embed_size, 101)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.cnn(images)
        outputs = self.lstm(features, captions, lengths)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        features = self.cnn(image).unsqueeze(0)
        result_caption = self.lstm.sample(features)[0]
        
        return [vocabulary.itos[str(idx.item())] for idx in result_caption]


class CaptionWithResnet18AndLstm(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithResnet18AndLstm, self).__init__()
        self.cnn = Resnet(embed_size, 18)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.cnn(images)
        outputs = self.lstm(features, captions, lengths)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        features = self.cnn(image).unsqueeze(0)
        result_caption = self.lstm.sample(features)[0]
        
        return [vocabulary.itos[str(idx.item())] for idx in result_caption]
