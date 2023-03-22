##############################################################################################
# This is a playground for writing experiments that can then be moved into the main codebase #
##############################################################################################


###########
# Imports #
###########
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from constants import Constants as const
from factories.data_factory import get_data
from models.components.vision.object_detectors import FasterRcnnResNet101BoundingBoxes, Detector

from graphs.spatial_graph_generator import generate_spatial_graphs_on_flickr8k
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


#####################
# SCRATCH FUNCTIONS #
#####################


def docs_test(img_name):
    img = read_image(f"/import/gameai-01/eey362/datasets/flickr8k/images/{img_name}")
    img = img.to(const.DEVICE)

    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
    model.to('cuda')
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4)
    im = to_pil_image(box.detach())
    im.save(f"{img_name}-test.png")


def obj_tester():
    train_loader, val_loader, train_dataset, val_dataset, padding = get_data('flickr8k')
    model = FasterRcnnResNet101BoundingBoxes(256).to(const.DEVICE)

    idx = 0
    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        if idx==10:break
        print(f"Processing {train_loader.dataset.dataset.imgs[train_loader.dataset.indices[idx]]}")
        images = data[0].to(const.DEVICE)
        predictions = model(images)
        
        for prediction in predictions:
            weights = model.faster_rcnn_weights
            labels = [weights.meta["categories"][i] for i in prediction["labels"]]
            for i in range(len(labels)):
                print(f"\tPredicted label: {labels[i]} with confidence: {prediction['scores'][i]}")
            print()
        print()
        idx += 1


########
# MAIN #
########
def main():
    # obj_tester()
    # docs_test('3055716848_b253324afc.jpg')
    # docs_test('3354883962_170d19bfe4.jpg')

    generate_spatial_graphs_on_flickr8k()
    


if __name__ == '__main__':
    const.DATASET = 'flickr8k'
    const.ROOT = "/import/gameai-01/eey362/datasets/flickr8k/images"
    const.ANNOTATIONS = "/import/gameai-01/eey362/datasets/flickr8k/captions.txt"
    const.TALK_FILE = "/homes/hps01/image-captioning/datasets/flickrtalk.json"
    const.BATCH_SIZE = 1
    const.NUM_WORKERS = 0
    const.PIN_MEMORY = False
    const.SHUFFLE = False
    
    # main()
    

    generate_spatial_graphs_on_flickr8k()
    data = torch.load('saved_models/spatial_graphs_flickr8k.pt')
    print()

