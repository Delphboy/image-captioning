import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                          fasterrcnn_resnet50_fpn_v2)
from models.components.vision.encoders import Resnet

class FasterRcnnResNet101BoundingBoxes(nn.Module):
    def __init__(self, 
                 embedding_size=256):
        super(FasterRcnnResNet101BoundingBoxes, self).__init__()
        self.embedding_size = embedding_size

        self.faster_rcnn_weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=self.faster_rcnn_weights, 
                                                      box_score_thresh=0.7)

        self.resnet_101 = Resnet(self.embedding_size, size=101)
        self.resnet_101.eval()


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
            debug_labels = [self.faster_rcnn_weights.meta["categories"][i] for i in prediction["labels"]]
            if (len(prediction['boxes']) == 0):
                # Treat the whole image as a single node
                print('Generated single node for image')
                prediction['boxes'] = torch.tensor([[0, 0, imgs[i].shape[0], imgs[i].shape[1]]])
            
            # TODO: Can we improve the cropping?
            for box in prediction['boxes']:
                xmin, ymin, xmax, ymax = box
                height = int(ymax) - int(ymin)
                width = int(xmax) - int(xmin)

                cropped_img = F.crop(imgs[i], int(ymin), int(xmin), height, width)

                # Resize the image to 299x299
                cropped_img = F.resize(cropped_img, (299, 299))
                cropped_imgs.append(cropped_img)

            # Get the resnet features for the cropped image
            crops = torch.stack(cropped_imgs)
            prediction['features'] = self.resnet_101(crops)

        return predictions




class Detector():
    def __init__(self, embedding_size) -> None:
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights, box_score_thresh=0.7)
        self.model.to('cuda')
        self.model.eval()
    

    def _detect_objects(self, imgs):
        # Step 2: Initialize the inference transforms
        preprocess = self.weights.transforms()

        # Step 3: Apply inference preprocessing transforms
        batch = preprocess(imgs)
        
        # Step 4: Use the model and visualize the prediction
        prediction = self.model(batch)[0]

        return prediction