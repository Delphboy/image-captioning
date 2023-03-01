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
from datasets.flickr import Flickr8kDataset
from factories.data_factory import get_coco_data, get_flickr8k_data
from models.components.vision.cnn import FasterRcnnResNet101BoundingBoxes

from graphs.graph_generators import SpatialGraphGenerator
from utils.precompute_graphs import precompute_flickr_spatial
#####################
# SCRATCH FUNCTIONS #
#####################






########
# MAIN #
########
def main():
    precompute_flickr_spatial(256)
    
    


if __name__ == '__main__':
    main()