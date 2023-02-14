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

from constants import Constants as const
from models.components.vision.cnn import FasterRcnnResNet101BoundingBoxes

#####################
# SCRATCH FUNCTIONS #
#####################






########
# MAIN #
########
def main():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = Flickr8kDataset(
        root_dir="/homes/hps01/flickr8k/images",
        captions_file="/homes/hps01/flickr8k/captions.txt",
        transform=transform,
        freq_threshold=5
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        # collate_fn=Flickr8kBatcher(pad_idx=dataset.vocab.stoi["<PAD>"]),
    )

    model = FasterRcnnResNet101BoundingBoxes(256).to(const.DEVICE)
    model.eval()


    images_processed = 0
    bad_images = 0

    for idx, (imgs) in tqdm(enumerate(loader), total=len(loader), leave=False):
        images_processed += 1
        images = imgs.to(const.DEVICE)
        outputs = model(images)

        if len(outputs[0]['boxes']) == 0:
            bad_images += 1

    print(f"{bad_images}/{images_processed} ({(bad_images / images_processed) * 100}%) bad images.")
    


if __name__ == '__main__':
    main()