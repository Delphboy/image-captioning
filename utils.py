import os
from datetime import datetime as dt
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from constants import Constants as const


def save_model_for_inference(model: nn.Module,
               save_loc: str='saved_models/',
               save_name: str=f"{dt.now().strftime('%d-%m-%Y-%H-%M')}.pth") -> None:
    path = os.path.join(os.getcwd(), save_loc, save_name)
    torch.save(model.state_dict(), path)


def load_model_for_inference(model: nn.Module, # A blank, empty model of the type being loaded
                             save_name: str,
                             save_loc: str='saved_models/') -> nn.Module:
    path = os.path.join(os.getcwd(), save_loc, save_name)

    model.load_state_dict(torch.load(path))
    model.to(device=const.DEVICE)
    model.eval()

    return model


def save_model_for_training(model: nn.Module,
                            optimiser: optim.Optimizer,
                            epoch: int,
                            loss: Any,
                            save_loc: str='saved_models/',
                            save_name: str=f"{dt.now().strftime('%d-%m-%Y-%H-%M')}") -> None:
    path = os.path.join(os.getcwd(), save_loc, save_name)
    save_dict = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }
    torch.save(save_dict, path)


def load_model_for_training(model: nn.Module, 
                            optimiser: optim.Optimizer, 
                            save_name: str, 
                            save_loc: str='saved_models/') -> nn.Module:
    path = os.path.join(save_loc, save_name)
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimiser, epoch, loss