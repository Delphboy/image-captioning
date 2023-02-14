import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.autograd import Variable
from torchvision.models import (Inception_V3_Weights, ResNet18_Weights,
                                ResNet101_Weights, ResNet152_Weights)
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)


class FasterRcnnResNet101BoundingBoxes(nn.Module):
    def __init__(self, embedding_size):
        super(FasterRcnnResNet101BoundingBoxes, self).__init__()
        self.embedding_size = embedding_size

        self.faster_rcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=self.faster_rcnn_weights, 
                                                      box_score_thresh=0.5)
        # self.faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=self.faster_rcnn_weights)
        self.resnet_101 = Resnet101(self.embedding_size)
        self.resnet_101 = self.resnet_101.eval()


    def _detect_objects(self, imgs):
        self.faster_rcnn.eval()
        preprocess = self.faster_rcnn_weights.transforms()
        preprocessed = preprocess(imgs)
        predictions = self.faster_rcnn(preprocessed)

        return predictions

    
    def forward(self, imgs):
        predictions = self._detect_objects(imgs)

        # TODO: Make the bounding box feature extraction faster! Consider creating a batch from the bounding boxes and giving them to RESNET in one go
        for i, prediction in enumerate(predictions):
            features = []
            if (len(prediction['boxes']) == 0):
                # Treat the whole image as a single node
                print('Generated single node for image')
                prediction['boxes'] = torch.tensor([[0, 0, imgs[i].shape[0], imgs[i].shape[1]]])
            
            for box in prediction['boxes']:
                # Crop the image to the bounding box
                top, left, bottom, right = box
                cropped_img = F.crop(imgs[i], int(top), int(left), int(bottom - top), int(right - left))
                
                # Resize the image to 299x299
                cropped_img = F.resize(cropped_img, (299, 299))
                
                # Get the resnet features for the cropped image
                unsqueeze = cropped_img.unsqueeze(0)
                resnet_features = self.resnet_101(unsqueeze)
                features.append(resnet_features)
            prediction['features'] = torch.stack(features)
            
        return predictions



class Resnet152(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Resnet152, self).__init__()
        resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-1] # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class Resnet101(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-101 and replace top fc layer."""
        super(Resnet101, self).__init__()
        resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-1] # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)


    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features
    

class Resnet18(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-18 and replace top fc layer."""
        super(Resnet18, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1] # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class InceptionV3(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(InceptionV3, self).__init__()
        self.train_CNN = train_CNN
        
        self.inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return_features = self.dropout(self.relu(features[0]))

        return return_features
