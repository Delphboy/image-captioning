from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import Constants as const
from utils.helper_functions import plot_training_loss, plot_training_and_val_loss


def train(model: nn.Module,
          optimiser: optim.Optimizer,
          scheduler: optim.lr_scheduler.ReduceLROnPlateau,
          loss_function: Any,
          train_data_loader: DataLoader,
          val_data_loader: DataLoader,
          epoch_count: int=10):
    
    training_loss_vals =  []
    val_loss_vals = []
    model.train()

    for epoch in range(1, epoch_count + 1):
        epoch_loss= []
        for idx, data in tqdm(enumerate(train_data_loader), total=len(train_data_loader), leave=False):
            # print(f"Processing {data_loader.dataset.dataset.imgs[data_loader.dataset.indices[idx]]}")
            images = data[0].to(const.DEVICE)
            targets = data[1].to(const.DEVICE)
            
            optimiser.zero_grad()
            # model.zero_grad()
            
            if const.IS_GRAPH_MODEL:
                graphs = data[3]
                graphs[0].cuda()
                graphs[1].cuda()
                logits = model(graphs, targets[:,:-1])
            else: 
                logits = model(images, targets[:,:-1])
            
            loss = loss_function(logits.permute(0, 2, 1), targets)
            loss.backward()
            optimiser.step()

            epoch_loss.append(loss.item())

        epoch_loss = sum(epoch_loss)/len(epoch_loss)
        scheduler.step(epoch_loss)

        print(f"Loss for epoch {epoch}/{epoch_count} | {epoch_loss}")
        training_loss_vals.append(epoch_loss)

        if epoch == 1 or epoch % 5 == 0:
            val_loss = evaluate(model, val_data_loader)
            val_loss_vals.append([epoch, val_loss])
    
    # plot_training_loss(np.linspace(1, epoch_count, epoch_count).astype(int), training_loss_vals)
    plot_training_and_val_loss(np.linspace(1, epoch_count, epoch_count).astype(int), training_loss_vals, val_loss_vals)

    return model, epoch, loss


@torch.no_grad()
def evaluate(model: nn.Module, 
             data_loader: DataLoader):
    model.eval()
    
    losses = []
    for idx, data in enumerate(data_loader):
        images = data[0].to(const.DEVICE)
        targets = data[1].to(const.DEVICE)
        
        if const.IS_GRAPH_MODEL:
            graphs = data[3]
            graphs[0].cuda()
            graphs[1].cuda()
            logits = model(graphs, targets[:,:-1])
        else: 
            logits = model(images, targets[:,:-1])
        
        loss = F.cross_entropy(logits.permute(0, 2, 1), 
                            targets, 
                            ignore_index=data_loader.dataset.vocab.stoi["<PAD>"])

        losses.append(loss.item())

    avg_loss = sum(losses)/len(losses)
    print(f"Validation loss: {avg_loss}")
    model.train()
    return avg_loss

