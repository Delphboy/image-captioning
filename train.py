import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import Constants as const


def plot_training_loss(epochs, loss):
    plt.plot(epochs, loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    plt.savefig(f'saves/loss_charts/loss-{const.MODEL_SAVE_NAME}-{now_str}.png')


def train(model: nn.Module,
          optimiser: optim.Optimizer,
          loss_function: Any,
          data_loader: DataLoader,
          epoch_count: int=10):
    
    loss_vals =  []
    model.train()

    for epoch in range(epoch_count):
        epoch_loss= []
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), leave=False):
            optimiser.zero_grad()

            # print(f"Processing {data_loader.dataset.dataset.imgs[data_loader.dataset.indices[idx]]}")
            images = data[0].to(const.DEVICE)
            captions = data[1].to(const.DEVICE)
            
            if const.IS_GRAPH_MODEL:
                graphs = data[3]
                graphs.cuda()
                outputs = model(images, captions, graphs)
            else: 
                outputs = model(images, captions[:,:-1])
            
            loss = loss_function(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            model.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss.append(loss.item())

        epoch_loss = sum(epoch_loss)/len(epoch_loss)
        print(f"Loss for epoch {epoch+1}/{epoch_count} | {epoch_loss}")
        loss_vals.append(epoch_loss)

    
    
    plot_training_loss(np.linspace(1, epoch_count, epoch_count).astype(int), loss_vals)

    return model, epoch, loss

