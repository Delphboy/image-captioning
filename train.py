from typing import Any

import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from constants import Constants as const
from their_code.utils import print_examples


def train(model: nn.Module,
          optimiser: optim.Optimizer,
          loss_function: Any,
          data_loader: DataLoader,
          epoch_count: int=10):
    
    model.train()

    for epoch in range(epoch_count):
        for idx, (imgs, captions, lengths) in enumerate(data_loader):
            images = imgs.to(const.DEVICE)
            captions = captions.to(const.DEVICE)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            outputs = model(images, captions, lengths)
            
            loss = loss_function(outputs, targets)
            model.zero_grad()
            loss.backward()
            
            optimiser.step()

            if idx == 0:
                model.eval()                
                print_examples(model, const.DEVICE, data_loader.dataset.dataset)
                model.train()
        
            print(f"Loss for epoch {epoch+1}/{epoch_count} | {idx+1}/{len(data_loader)} | {loss}")
            
    
    return model, epoch, loss