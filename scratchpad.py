##############################################################################################
# This is a playground for writing experiments that can then be moved into the main codebase #
##############################################################################################


###########
# Imports #
###########

# from torchvision.io.image import read_image
# from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
#                                           fasterrcnn_resnet50_fpn_v2)
# from torchvision.transforms.functional import to_pil_image
# from torchvision.utils import draw_bounding_boxes
# from tqdm import tqdm

# from models.components.vision.object_detectors import FasterRcnnResNet101BoundingBoxes
from constants import Constants as const
from datasets.flickr import Flickr8kDataset
from factories.data_factory import get_data
# from factories.data_factory import get_data
from graphs.spatial_graph_generator import generate_spatial_graphs_on_coco_karpathy, generate_spatial_graphs_on_flickr8k
from graphs.semantic_graph_generator import generate_semantic_graphs_on_coco_karpathy

#####################
# SCRATCH FUNCTIONS #
#####################


# def docs_test(img_name):
#     img = read_image(f"/import/gameai-01/eey362/datasets/flickr8k/images/{img_name}")
#     img = img.to(const.DEVICE)

#     # Step 1: Initialize model with the best available weights
#     weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
#     model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.7)
#     model.to('cuda')
#     model.eval()

#     # Step 2: Initialize the inference transforms
#     preprocess = weights.transforms()

#     # Step 3: Apply inference preprocessing transforms
#     batch = [preprocess(img)]

#     # Step 4: Use the model and visualize the prediction
#     prediction = model(batch)[0]
#     labels = [weights.meta["categories"][i] for i in prediction["labels"]]
#     box = draw_bounding_boxes(img, boxes=prediction["boxes"],
#                             labels=labels,
#                             colors="red",
#                             width=4)
#     im = to_pil_image(box.detach())
#     im.save(f"{img_name}-test.png")


# def obj_tester():
#     train_loader, val_loader, train_dataset, val_dataset, padding = get_data('flickr8k')
#     model = FasterRcnnResNet101BoundingBoxes(256).to(const.DEVICE)

#     idx = 0
#     for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
#         if idx==10:break
#         print(f"Processing {train_loader.dataset.dataset.imgs[train_loader.dataset.indices[idx]]}")
#         images = data[0].to(const.DEVICE)
#         predictions = model(images)
        
#         for prediction in predictions:
#             weights = model.faster_rcnn_weights
#             labels = [weights.meta["categories"][i] for i in prediction["labels"]]
#             for i in range(len(labels)):
#                 print(f"\tPredicted label: {labels[i]} with confidence: {prediction['scores'][i]}")
#             print()
#         print()
#         idx += 1


########
# MAIN #
########
def main():
    # generate_spatial_graphs_on_flickr8k(False)
    # generate_spatial_graphs_on_coco_karpathy(split='test', 
    #                                          include_relationship_weights=False)
    
    generate_semantic_graphs_on_coco_karpathy(split='val')


if __name__ == '__main__':
    # const.DATASET = 'coco_karpathy'
    # const.ROOT = "/import/gameai-01/eey362/datasets/coco/images"
    # const.ANNOTATIONS = "/homes/hps01/image-captioning/datasets/splits/dataset_coco.json"
    # const.TALK_FILE = "/homes/hps01/image-captioning/datasets/talk_files/cocotalk.json"

    # const.IS_GRAPH_MODEL = True
    # const.DATASET = 'flickr8k'
    # const.ROOT = "/import/gameai-01/eey362/datasets/flickr8k/images"
    # const.ANNOTATIONS = "/homes/hps01/image-captioning/datasets/splits/dataset_flickr8k.json"
    # const.TALK_FILE = "/homes/hps01/image-captioning/datasets/talk_files/flickrtalk.json"
    
    # const.PRECOMPUTED_SPATIAL_GRAPHS = {
    #     "train": "/import/gameai-01/eey362/datasets/flickr8k/precomputed_graphs/flickr8k_spatial_train.pt",
    #     "val": "/import/gameai-01/eey362/datasets/flickr8k/precomputed_graphs/flickr8k_spatial_val.pt",
    #     "test": "/import/gameai-01/eey362/datasets/flickr8k/precomputed_graphs/flickr8k_spatial_test.pt"
    # }


    # # main()

    # train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, pad_idx = get_data(const.DATASET)
    # print(len(train_dataset))
    # print(len(train_loader))
    # print("-"*10)
    # print(len(val_dataset))
    # print(len(val_loader))
    # print("-"*10)
    # print(len(test_dataset))
    # print(len(test_loader))
    

    # for idx, data in enumerate(val_loader):
    #     print(idx)

    # generate_spatial_graphs_on_flickr8k('val', True)
    generate_spatial_graphs_on_coco_karpathy('val', True)