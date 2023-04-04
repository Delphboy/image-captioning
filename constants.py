from dataclasses import dataclass
import torch
import torchvision.transforms as transforms
import spacy

@dataclass
class Constants:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SPACY_ENG = spacy.load("en_core_web_sm")
    STANDARD_TRANSFORM = transforms.Compose(
        [
            transforms.functional.convert_image_dtype,
            transforms.Resize((356, 356)),
            # transforms.RandomCrop((299, 299)),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # These settings are loaded in from the config file in main.py
    MODEL = ""
    MODEL_SAVE_NAME = ""
    REGIME = "train_and_test"
    DATASET = "flickr8k"
    ROOT = ""
    ANNOTATIONS = ""
    TALK_FILE = ""
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    SHUFFLE = False
    PIN_MEMORY = True
    LEARNING_RATE = 3e-4
    EPOCHS = 1
    IS_GRAPH_MODEL = False
    PRECOMPUTED_SPATIAL_GRAPHS = ""