import os
from datetime import datetime as dt
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from constants import Constants as const

def save_model_checkpoint(model: nn.Module,
                            optimiser: optim.Optimizer,
                            epoch: int,
                            loss: Any,
                            save_loc: str='saves/saved_models/',
                            save_name: str=f"{dt.now().strftime('%d-%m-%Y-%H-%M')}.pth") -> None:
    if not save_name.endswith('.pth'):
        save_name = f"{save_name}.pth"
    
    path = os.path.join(os.getcwd(), save_loc, save_name)
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }
    torch.save(checkpoint, path)


def load_model(model: nn.Module, 
                optimiser: Optional[optim.Optimizer], 
                save_name: str, 
                save_loc: str='saves/saved_models/') -> Tuple[nn.Module, 
                                                        Optional[optim.Optimizer], 
                                                        Optional[Any], 
                                                        Optional[Any]]:
    path = os.path.join(save_loc, f"{save_name}.pth")
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(const.DEVICE)

    if optimiser is None:
        model.eval()
        return model

    model.train()
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimiser, epoch, loss
