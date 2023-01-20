from dataclasses import dataclass
import torch

@dataclass
class Constants:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')