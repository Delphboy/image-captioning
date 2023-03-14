from typing import Any
import numpy as np

import datetime
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import Constants as const
import matplotlib.pyplot as plt

from graphs.spatial_graph_generator import SpatialGraphGenerator

def plot_training_loss(epochs, loss):
    plt.plot(epochs, loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    plt.savefig(f'saved_models/loss-{now_str}.png')


# TODO: Make into one train loop that figures out if it's a graph model or not

def train(model: nn.Module,
          optimiser: optim.Optimizer,
          loss_function: Any,
          data_loader: DataLoader,
          epoch_count: int=10):
    
    # model.train()

    loss_vals =  []
    for epoch in range(epoch_count):
        epoch_loss= []
        for idx, (imgs, captions, lengths) in enumerate(data_loader):
            images = imgs.to(const.DEVICE)
            captions = captions.to(const.DEVICE)
            
            lengths = lengths.to('cpu') # pack_padded_sequence requires lengths to be on cpu
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            outputs = model(images, captions, lengths)
            
            loss = loss_function(outputs, targets)
            model.zero_grad()
            loss.backward()

            epoch_loss.append(loss.item())
            
            optimiser.step()
        
        epoch_loss = sum(epoch_loss)/len(epoch_loss)
        print(f"Loss for epoch {epoch+1}/{epoch_count} | {epoch_loss}")
        loss_vals.append(epoch_loss)

    
    
    plot_training_loss(np.linspace(1, epoch_count, epoch_count).astype(int), loss_vals)

    return model, epoch, loss


def train_graph_model(model: nn.Module, 
                      optimiser: optim.Optimizer, 
                      loss_function: Any, 
                      data_loader: DataLoader, 
                      epoch_count: int=10):
       
    loss_vals =  []
    for epoch in range(epoch_count):
        epoch_loss= []
        for idx, (imgs, captions, lengths, graphs) in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
            images = imgs.to(const.DEVICE)
            captions = captions.to(const.DEVICE)
            graphs.cuda()
            
            lengths = lengths.to('cpu') # pack_padded_sequence requires lengths to be on cpu
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            outputs = model(images, captions, lengths, graphs)
            
            loss = loss_function(outputs, targets)
            model.zero_grad()
            loss.backward()

            epoch_loss.append(loss.item())
            
            optimiser.step()
        
        epoch_loss = sum(epoch_loss)/len(epoch_loss)
        print(f"Loss for epoch {epoch+1}/{epoch_count} | {epoch_loss}")
        loss_vals.append(epoch_loss)

    
    
    plot_training_loss(np.linspace(1, epoch_count, epoch_count).astype(int), loss_vals)

    return model, epoch, loss

