from typing import Any

from constants import Constants as const
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim


def train(model: nn.Module,
          optimiser: optim.Optimizer,
          loss_function: Any,
          data_loader: DataLoader,
          epoch_count: int=10):
    
    step = 0
    model.train()

    for epoch in range(epoch_count):
        for idx, (imgs, captions) in enumerate(data_loader):
            imgs = imgs.to(const.DEVICE)
            captions = captions.to(const.DEVICE)
            
            outputs = model(imgs, captions[:-1])
            loss = loss_function(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            step += 1

            optimiser.zero_grad()
            loss.backward(loss)
            optimiser.step()
        
        print(f"Loss for epoch {epoch}: {loss}")
    
    return model