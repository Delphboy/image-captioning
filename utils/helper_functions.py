import os
import datetime
import json
import numpy as np
from typing import List

import matplotlib.pyplot as plt
from PIL import Image
import torch

from constants import Constants as const
from factories.model_factory import MODELS


def parse_config_file(file_loc: str):
    with open(file_loc) as f:
        config = json.load(f)
    
    const.MODEL = config["model"]

    if const.MODEL not in MODELS.keys():
        raise ValueError(f"Model {const.MODEL} not in list of available models: {MODELS.keys()}")

    const.MODEL_SAVE_NAME = config["model_save_name"]

    const.REGIME = config["regime"]

    const.DATASET = config["dataset"]["name"]
    const.ROOT = config["dataset"]["root"]
    const.ANNOTATIONS = config["dataset"]["annotations"]
    const.TALK_FILE = config["dataset"]["talk_file"]
    
    const.BATCH_SIZE = config["training_parameters"]["batch_size"]
    const.NUM_WORKERS = config["training_parameters"]["num_workers"]
    const.SHUFFLE = config["training_parameters"]["shuffle"] == "True"
    const.PIN_MEMORY = config["training_parameters"]["pin_memory"] == "True"
    const.LEARNING_RATE = config["training_parameters"]["learning_rate"]
    const.EPOCHS = config["training_parameters"]["epochs"]
    const.IS_GRAPH_MODEL = config["is_graph_model"] == "True"

    if const.IS_GRAPH_MODEL:
        const.PRECOMPUTED_SPATIAL_GRAPHS = config["graph_configurations"]["precomputed_spatial_graphs"]
        const.PRECOMPUTED_SEMANTIC_GRAPHS = config["graph_configurations"]["precomputed_semantic_graphs"]


def get_test_image(location, transform=None):
    img = Image.open(location).convert("RGB")
    if transform is not None:
        img = transform(img)
        img = img.unsqueeze(0)
    return img


def caption_array_to_string(array: List[str]) -> str:
    caption = ""

    for i in range(1, len(array)):
        item = array[i]

        if item == "<EOS>": break

        # The captions.txt has a space before fullstops
        if item != '.':
            caption += f"{item} "
        else:
            caption += "."

    return caption


def plot_training_loss(epochs, loss):
    plt.plot(epochs, loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    plt.savefig(f'saves/loss_charts/loss-{const.MODEL_SAVE_NAME}-{now_str}.png')


def plot_training_and_val_loss(epochs, training_loss, val_loss):
    # plt.plot(epochs, training_loss)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")

    # now = datetime.datetime.now()
    # now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # plt.savefig(f'saves/loss_charts/loss-{const.MODEL_SAVE_NAME}-{now_str}.png')

    # Using matplotlib, plot a line graph with two lines
    # The y axis is the loss value
    # The x axis is the epoch number
    # epochs is an np array of epoch numbers
    # Training loss is an np array of training loss values, where the ith value is the loss for the ith epoch
    # val_loss is a list of [epoch, val_loss] pairs
    # Plot the training loss as a line
    plt.plot(epochs, training_loss, label="Training Loss")
    # Plot the validation loss as a line
    epochs, val_loss = zip(*val_loss)
        
    plt.plot(epochs, val_loss, label="Validation Loss")
    # Add a legend to the plot
    plt.legend()
    # Add a title to the plot
    plt.title("Training and Validation Loss")
    # Add x and y axis labels
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    plt.savefig(f'saves/loss_charts/val-train-loss-{const.MODEL_SAVE_NAME}-{now_str}.png')


def read_coco_karpathy_attributes(directory: str, 
                                  attribute_folder: str, 
                                  image_id: int) -> list:
    attribute_file_loc = os.path.join(directory, attribute_folder, f'{image_id}.npz')
    attribute = np.load(attribute_file_loc, allow_pickle=True, encoding='bytes')
    return attribute['feat']


def read_bottom_up_top_down_features(image_id: int) -> dict:
    attribute_file_loc = "../data/cocobu_att/"
    bbox_file_loc = "../data/cocobu_box/"
    image_data = {}
    
    attributes = np.load(os.path.join(const.ROOT, attribute_file_loc, f"{image_id}.npz"), 
                         allow_pickle=True, 
                         encoding='bytes')['feat']
    attributes = torch.from_numpy(attributes)
    bboxes = np.load(os.path.join(const.ROOT, bbox_file_loc, f"{image_id}.npy"), 
                     allow_pickle=True, 
                     encoding='bytes')
    bboxes = torch.from_numpy(bboxes)
    
    image_data['features'] = attributes
    image_data['boxes'] = bboxes
    
    return image_data

