import torch
import torch.nn as nn


class PoolEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features):
        x = torch.mean(input_features, dim=1).unsqueeze(1)
        return x


class NoneEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features):
        return input_features
