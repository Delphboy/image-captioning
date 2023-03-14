import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from models.components.vision.encoders import Resnet

class FasterRcnnResNet101BoundingBoxes(nn.Module):
    def __init__(self, embedding_size):
        super(FasterRcnnResNet101BoundingBoxes, self).__init__()
        self.embedding_size = embedding_size

        self.faster_rcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        # self.faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=self.faster_rcnn_weights, 
        #                                               box_score_thresh=0.5)
        self.faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=self.faster_rcnn_weights)

        self.resnet_101 = Resnet(self.embedding_size, size=101)
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
            cropped_imgs = []
            
            if (len(prediction['boxes']) == 0):
                # Treat the whole image as a single node
                print('Generated single node for image')
                prediction['boxes'] = torch.tensor([[0, 0, imgs[i].shape[0], imgs[i].shape[1]]])
            
            # TODO: Can we improve the cropping?
            for box in prediction['boxes']:
                # Crop the image to the bounding box (xmin, ymin, xmax, ymax)
                # top, left, bottom, right = box
                # cropped_img = F.crop(imgs[i], int(top), int(left), int(bottom - top), int(right - left))
                
                xmin, ymin, xmax, ymax = box
                height = int(ymax) - int(ymin)
                if height == 0: height = 1
                
                width = int(xmax) - int(xmin)
                if width == 0: width = 1 # TODO: This is a hack to fix the bug in the next line
                cropped_img = F.crop(imgs[i], int(ymin), int(xmin), height, width)

                # Resize the image to 299x299
                cropped_img = F.resize(cropped_img, (299, 299)) # BUG: RuntimeError: Input and output sizes should be greater than 0, but got input (H: 0, W: 113) output (H: 299, W: 299)
                cropped_imgs.append(cropped_img)

            # Get the resnet features for the cropped image
            crops = torch.stack(cropped_imgs)
            prediction['features'] = self.resnet_101(crops)

        return predictions
