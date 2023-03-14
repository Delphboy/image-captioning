from dataclasses import dataclass
import torch
import torchvision.transforms as transforms
import spacy

@dataclass
class Constants:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO: This should come from a config file
    SPACY_ENG = spacy.load("en_core_web_sm")
    FLICKR_ROOT = '/data/scratch/eey362/flickr8k/images' # TODO: These should come from a config file
    FLICKR_ANN = '/data/scratch/eey362/flickr8k/captions.txt'
    PRECOMPUTED_SPATIAL_GRAPHS = '/data/home/eey362/image-captioning/saved_models/flickr_spatial_graphs.pt'
    STANDARD_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )