import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (Inception_V3_Weights, ResNet18_Weights,
                                ResNet50_Weights, ResNet101_Weights, 
                                ResNet152_Weights)


class Resnet(nn.Module):
    def __init__(self, embed_size, size=101):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Resnet, self).__init__()
        
        resnet = None
        assert size in [18, 50, 101, 152]
        
        if size == 18:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif size == 50:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif size == 101:
            resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
       
        modules = list(resnet.children())[:-1] # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)


    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        if features.shape[0] > 1:
            features = self.bn(features)
        return features

   
class InceptionV3(nn.Module):
    def __init__(self, embed_size, is_cnn_training=False):
        super(InceptionV3, self).__init__()
        self.is_cnn_training = is_cnn_training
        
        self.inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = is_cnn_training

    def forward(self, images):
        features = self.inception(images)
        return_features = self.dropout(self.relu(features[0]))

        return return_features
