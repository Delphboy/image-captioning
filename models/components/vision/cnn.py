import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.models import ResNet101_Weights
from torch.autograd import Variable
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


# class FasterRcnn(nn.Module):
#     def __init__(self, embedding_size):
#         super().__init__()

#         self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
#         self.faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.9)
        
#         self.resnet = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_size)

#         self.dropout = nn.Dropout(0.5)
#         self.relu = nn.ReLU()
    

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         features = self.resnet(X)
#         for name, param in self.resnet.named_parameters():
#             param.requires_grad = False

#         return self.dropout(self.relu(features))


class InceptionV3(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(InceptionV3, self).__init__()
        self.train_CNN = train_CNN
        
        # FIXME: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
        # FIXME: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=Inception_V3_Weights.IMAGENET1K_V1`. You can also use `weights=Inception_V3_Weights.DEFAULT` to get the most up-to-date weights.
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features[0]))
