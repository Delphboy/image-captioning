import datetime
import json
from typing import List

import matplotlib.pyplot as plt
from PIL import Image

from constants import Constants as const
from factories.model_factory import MODELS


def parse_file(file_loc: str):
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
        const.PRECOMPUTED_SPATIAL_GRAPHS = config["graph_configurations"]["precomputed_spatial_graph_location"]


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

