import torch
import torch.nn as nn

from models.components.vision.cnn import InceptionV3, Resnet101, Resnet152, Resnet18, FasterRcnnResNet101BoundingBoxes
from models.components.language.lstm import Lstm

from graphs.graph_generators import SpatialGraphGenerator
from graphs.gnns import Gcn


class CaptionWithSpatialGraph(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithSpatialGraph, self).__init__()
        self.cnn = FasterRcnnResNet101BoundingBoxes(embed_size)
        self.gcn = Gcn(embed_size, embed_size)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)

        self.spaital_graph_generator = SpatialGraphGenerator()
    
    
    def forward(self, images, captions, lengths):
        # Get objects
        image_predictions = self.cnn(images)
        
        # Generate spatial graph
        spatial_graphs = self.spaital_graph_generator.generate_spatial_graph_for_batch(image_predictions)        
    
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
        
        return [vocabulary.itos[idx.item()] for idx in result_caption]



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
        self.cnn = Resnet152(embed_size)
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
        self.cnn = Resnet101(embed_size)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.cnn(images)
        outputs = self.lstm(features, captions, lengths)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        features = self.cnn(image).unsqueeze(0)
        result_caption = self.lstm.sample(features)[0]
        
        return [vocabulary.itos[idx.item()] for idx in result_caption]


class CaptionWithResnet18AndLstm(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithResnet18AndLstm, self).__init__()
        self.cnn = Resnet18(embed_size)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.cnn(images)
        outputs = self.lstm(features, captions, lengths)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        features = self.cnn(image).unsqueeze(0)
        result_caption = self.lstm.sample(features)[0]
        
        return [vocabulary.itos[str(idx.item())] for idx in result_caption]


class CaptionWithSpatialGraphAndLstm(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super(CaptionWithSpatialGraphAndLstm, self).__init__()
        self.graph_generator = SpatialGraphGenerator()
        
        self.gcn = Gcn(embed_size, embed_size)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)

        
    def forward(self, imgs, captions, lengths):

        # 1. Generate scene graphs for each graph in the batch (consider pre-computing these at a later date)
        G = self.graph_generator.generate_spatial_graph_for_batch(imgs)
    
        # 2. GCN on graph
        pre_x = G.get_nodes()
        edge_index = G.get_edges()
        convolved_graph = self.gcn(pre_x, edge_index)

        # 3. Mean pool graph nodes
        features = torch.mean(convolved_graph, dim=0)

        # 4. LSTM
        outputs = self.lstm(features, captions, lengths)

        return outputs