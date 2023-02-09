from dataclasses import dataclass
import torch
import spacy

@dataclass
class Constants:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SPACY_ENG = spacy.load("en_core_web_sm")
