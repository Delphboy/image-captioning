import os
import datetime
import json
import numpy as np
from typing import List, Optional

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


def caption_array_to_string(array: List[str],
                            is_scst:Optional[bool]=False) -> str:
    caption = ""
    for i in range(0, len(array)):
        item = array[i]

        if item == "<SOS>": continue
        if item == "<PAD>": continue
        if item == "<EOS>":
            if is_scst: caption += item + " "
            break
        caption += item + " "
    caption = caption.strip()

    return caption


def plot_training_loss(epochs, loss):
    plt.plot(epochs, loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_loc = os.path.join(os.getcwd(), f'saves/loss_charts/loss-{const.MODEL_SAVE_NAME}-{now_str}.png')
    plt.savefig(save_loc)


def plot_training(training_loss,
                  val_loss):
    plt.figure(figsize=(10, 10))
    epochs, training_loss = zip(*training_loss)
    plt.plot(epochs, training_loss, label="Training Loss")
    
    epochs, val_loss = zip(*val_loss)    
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.axvline(x=const.EPOCHS, color='r', linestyle='--', label="End of XE Training")

    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_loc = os.path.join(os.getcwd(), f'saves/loss_charts/training-metrics-{const.MODEL_SAVE_NAME}-{now_str}.png')
    plt.savefig(f'{save_loc}')


def plot_metrics(performance_metrics):
    plt.figure(figsize=(10, 10))
    for metric in performance_metrics[0][1].keys():
        epochs, metric_values = zip(*[(pm[0], pm[1][metric]*100) for pm in performance_metrics])
        plt.plot(epochs, metric_values, label=f"Val {metric}")

    plt.axvline(x=const.EPOCHS, color='r', linestyle='--', label="End of XE Training")

    plt.legend()
    plt.title("Validation Set Metrics")
    plt.xlabel("Epochs")
    plt.ylabel("Value")

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_loc = os.path.join(os.getcwd(), f'saves/loss_charts/validation-performance-{const.MODEL_SAVE_NAME}-{now_str}.png')
    plt.savefig(f'{save_loc}')
    


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

