##############################################################################################
# This is a playground for writing experiments that can then be moved into the main codebase #
##############################################################################################


###########
# Imports #
###########
import json
import os
import pickle
from collections import Counter
from turtle import pd
from typing import Any, Callable, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.optim as optim
from constants import Constants as const
from datasets.flickr import Flickr8kDataset
from factories.data_factory import get_coco_data, get_flickr8k_data, get_flickr8k_data_with_spatial_graphs
from factories.model_factory import get_model
from models.components.vision.object_detectors import FasterRcnnResNet101BoundingBoxes
from train import train_graph_model

from graphs.spatial_graph_generator import SpatialGraphGenerator
from utils.precompute_graphs import precompute_flickr_spatial
#####################
# SCRATCH FUNCTIONS #
#####################






########
# MAIN #
########
def main():
    trainer, tester, dataset = get_flickr8k_data_with_spatial_graphs(
        root_folder=const.FLICKR_ROOT,
        annotation_file=const.FLICKR_ANN,
        transform=const.STANDARD_TRANSFORM,
        graph_dir='/homes/hps01/image-captioning/saved_models/flickr_spatial_graphs.pt',
        train_ratio=0.8,
        batch_size=8,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )
    
    vocab_size = len(dataset.vocab)

    model = get_model('spatialgcn', vocab_size)

    learning_rate = 3e-4
    epochs=100
    cross_entropy = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    adam_optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    trained, epoch, loss = train_graph_model(model=model, 
                                        optimiser=adam_optimiser, 
                                        loss_function=cross_entropy, 
                                        data_loader=trainer, 
                                        epoch_count=epochs)


if __name__ == '__main__':
    # main()
    print('hey')