from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from constants import Constants as const
from models.base_captioner import BaseCaptioner
from utils.helper_functions import plot_training_and_val_loss, caption_array_to_string
from tqdm import tqdm
from pycocoevalcap.eval import Cider


def train_supervised(model: nn.Module,
                     optimiser: optim.Optimizer,
                     scheduler: optim.lr_scheduler._LRScheduler,
                     loss_function: Any,
                     train_data_loader: DataLoader,
                     val_data_loader: DataLoader,
                     epoch_count: int=10):
    
    training_loss_vals =  []
    val_loss_vals = []
    EARLY_STOP_THRESHOLD = 5
    early_stopping_count = 0
    avg_epoch_loss = 10
    model.train()

    for epoch in range(1, epoch_count+1):
        epoch_loss= []
        wrapped_loader = tqdm(enumerate(train_data_loader), desc=f"Last epoch's loss: {avg_epoch_loss:.4f}")
        for idx, data in wrapped_loader:
            # print(f"Processing {train_data_loader.dataset.ids[idx]}")
            images = data[0].to(const.DEVICE, non_blocking=True)
            targets = data[1].to(const.DEVICE, non_blocking=True)
            
            optimiser.zero_grad()
            # model.zero_grad()
            if const.IS_GRAPH_MODEL:
                graphs = data[3]
                graphs[0].to(const.DEVICE, non_blocking=True)
                graphs[0].edge_index.to('cpu')
                graphs[1].to(const.DEVICE, non_blocking=True)
                graphs[1].edge_index.to('cpu')
                logits = model(graphs, targets[:,:-1])
            else: 
                logits = model(images, targets[:,:-1])
            
            loss = loss_function(logits.permute(0, 2, 1), targets)
            loss.backward()
            optimiser.step()

            epoch_loss.append(loss.item())

        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        scheduler.step()

        print(f"Loss for epoch {epoch}/{epoch_count} | {avg_epoch_loss}")
        training_loss_vals.append(avg_epoch_loss)
        wrapped_loader.set_description(f"Last epoch's loss: {avg_epoch_loss:.4f}")

        if epoch == 1 or epoch % 5 == 0:
            val_loss = evaluate(model, val_data_loader)
            val_loss_vals.append([epoch, val_loss])
        
        if avg_epoch_loss > training_loss_vals[-1] or val_loss > val_loss_vals[-1][1]:
            early_stopping_count += 1
            print("Early stopping count increased")
            if early_stopping_count > EARLY_STOP_THRESHOLD:
                print(f"Early stopping after {epoch} epochs")
                return model, epoch, avg_epoch_loss
        else:
            early_stopping_count = 0
    
    # plot_training_loss(np.linspace(1, epoch_count, epoch_count).astype(int), training_loss_vals)
    plot_training_and_val_loss(np.linspace(1, epoch_count, epoch_count).astype(int), training_loss_vals, val_loss_vals)
    print(f"Training complete after {epoch_count} epochs")
    print(f"Final training loss: {training_loss_vals[-1]}")
    print(f"Final validation loss: {val_loss_vals[-1][1]}")
    print(f"Final learning rate: {optimiser.param_groups[0]['lr']}")
    return model, epoch, avg_epoch_loss


def train_self_critical(model: BaseCaptioner,
                        optimiser: optim.Optimizer,
                        train_data: DataLoader,
                        val_data: DataLoader,
                        epoch_count: int=10):
    
    cider = Cider()
    model.train()

    for epoch in range(1, epoch_count+1):
        running_loss = 0.0
        for idx, data in enumerate(train_data):
            images = data[0].to(const.DEVICE, non_blocking=True)
            targets = data[1].to(const.DEVICE, non_blocking=True)

            optimiser.zero_grad()

            test_candidates, probabilities = model.greedy_caption(images, train_data.dataset.vocab)
            probabilities = torch.tensor(probabilities).to(const.DEVICE, non_blocking=True)

            # Get the ground truth captions
            target_list = [targets[i].tolist() for i in range(targets.shape[0])]
            ground_truths = []
            for target in target_list:
                words = [train_data.dataset.vocab.itos[str(i)] for i in target]
                ground_truths.append(caption_array_to_string(words))

            # Create the dictionaries for the cider score
            test_candidate_dict = {}
            for i, candidate in enumerate(test_candidates):
                test_candidate_dict[i] = [" ".join(candidate)]
            
            ground_truth_dict = {}
            for i, ground_truth in enumerate(ground_truths):
                ground_truth_dict[i] = ["<SOS> " + ground_truth + " <EOS>"]

            # Calculate the cider score
            rewards = cider.compute_score(ground_truth_dict, test_candidate_dict)[1]
            rewards = torch.tensor(rewards, requires_grad=True).to(const.DEVICE, non_blocking=True)

            # Calculate the loss
            log_probs = torch.log(probabilities)
            reward_baseline = torch.mean(rewards, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (rewards - reward_baseline)
            loss = loss.mean()

            # updates
            loss.backward()
            optimiser.step()
        
            running_loss += loss.item()
        
        print(f"Epoch: {epoch}, Running Loss: {running_loss/len(train_data)}")
    
    return model, epoch, running_loss



@torch.no_grad()
def evaluate(model: nn.Module, 
             data_loader: DataLoader):
    model.eval()
    
    losses = []
    enumerator = tqdm(enumerate(data_loader))
    for idx, data in enumerator:
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

