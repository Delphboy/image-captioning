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
    MODEL = None
    MODEL_SAVE_NAME = None
    REGIME = None
    DATASET = None
    ROOT = None
    ANNOTATIONS = None
    TALK_FILE = None
    BATCH_SIZE = None
    NUM_WORKERS = None
    SHUFFLE = False
    PIN_MEMORY = True
    LEARNING_RATE = None
    EPOCHS = None
    IS_GRAPH_MODEL = None
    PRECOMPUTED_SPATIAL_GRAPHS = None