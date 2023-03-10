from dataclasses import dataclass
import torch
import torchvision.transforms as transforms
import spacy

@dataclass
class Constants:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SPACY_ENG = spacy.load("en_core_web_sm")
    FLICKR_ROOT = '/import/gameai-01/eey362/datasets/flickr8k/images' # TODO: These should come from a config file
    FLICKR_ANN = '/import/gameai-01/eey362/datasets/flickr8k/captions.txt'
    STANDARD_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )