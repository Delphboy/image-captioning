from typing import Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch_geometric.data import Data, Batch

from constants import Constants as const
from models.base_captioner import BaseCaptioner
from utils.helper_functions import caption_array_to_string
from tqdm import tqdm
from pycocoevalcap.eval import Cider, Rouge
from eval import evaluate_caption_model

train_loss_vals =  []
val_loss_vals = []
val_performance_vals = []


def train_supervised(model: nn.Module,
                     optimiser: optim.Optimizer,
                     scheduler: optim.lr_scheduler._LRScheduler,
                     loss_function: Any,
                     train_data_loader: DataLoader,
                     val_data_loader: DataLoader,
                     epoch_count: int=10):
    
    EARLY_STOP_THRESHOLD = 5
    early_stopping_count = 0
    avg_epoch_loss = 10
    model.train()

    for epoch in range(1, epoch_count+1):
        epoch_loss= []
        wrapped_loader = tqdm(enumerate(train_data_loader), desc=f"Last epoch's loss: {avg_epoch_loss:.4f}")
        for idx, data in wrapped_loader:
            images = data[0].to(const.DEVICE, non_blocking=True)

            targets = data[1].to(const.DEVICE, non_blocking=True)
            targets = targets[0:-1:5,:]
            
            lengths = data[2].to(const.DEVICE, non_blocking=True)
            lengths = lengths[0:-1:5]

            # Remove excess padding
            targets = targets[:,:max(lengths)] 
            
            optimiser.zero_grad()
            if const.IS_GRAPH_MODEL:
                graphs = data[3]
                graphs[0].to(const.DEVICE, non_blocking=True)
                graphs[0].edge_index.to('cpu')
                graphs[1].to(const.DEVICE, non_blocking=True)
                graphs[1].edge_index.to('cpu')
                logits = model(graphs, targets[:,:-1], lengths)
            else: 
                logits = model(images, targets[:,:-1])
            
            loss = loss_function(logits.permute(0, 2, 1), 
                                 targets[:,1:])
            loss.backward()
            optimiser.step()

            epoch_loss.append(loss.item())
        
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        scheduler.step()

        print(f"Loss for epoch {epoch}/{epoch_count} | {avg_epoch_loss}")
        train_loss_vals.append([epoch, avg_epoch_loss])
        wrapped_loader.set_description(f"Last epoch's loss: {avg_epoch_loss:.4f}")

        # Evaluate the model on the validation set
        if epoch == 1 or epoch % 10 == 0:
            val_loss = evaluate(model, val_data_loader)
            val_loss_vals.append([epoch, val_loss])
            global_results, _ = evaluate_caption_model(model, val_data_loader.dataset)
            val_performance_vals.append([epoch, global_results])
            model.train()

        
        if avg_epoch_loss > train_loss_vals[-1][1] or val_loss > val_loss_vals[-1][1]:
            early_stopping_count += 1
            print("Early stopping count increased")
            if early_stopping_count > EARLY_STOP_THRESHOLD:
                print(f"Early stopping after {epoch} epochs")
                return model, epoch, avg_epoch_loss
        else:
            early_stopping_count = 0
    
    print(f"Training complete after {epoch_count} epochs of Cross Entropy Loss Training")
    print(f"Final training loss: {train_loss_vals[-1]}")
    print(f"Final validation loss: {val_loss_vals[-1][1]}")
    print(f"Final learning rate: {optimiser.param_groups[0]['lr']}")
    
    return model, epoch, avg_epoch_loss


def train_self_critical(model: BaseCaptioner,
                        optimiser: optim.Optimizer,
                        train_data_loader: DataLoader,
                        val_data_loader: DataLoader,
                        epoch_count: int=10):
    vocab = train_data_loader.dataset.vocab
    convert = lambda idxs: [vocab.itos[f"{int(idx)}"] for idx in idxs]

    model.train()

    for epoch in range(1, epoch_count+1):
        running_loss = 0.0
        for idx, data in enumerate(train_data_loader):
            images = data[0].to(const.DEVICE, non_blocking=True)
            lengths = data[2].to(const.DEVICE, non_blocking=True)
            
            targets = data[1].to(const.DEVICE, non_blocking=True)
            
            graphs = data[3]
            graphs[0].to(const.DEVICE, non_blocking=True)
            graphs[0].edge_index.to('cpu')
            graphs[1].to(const.DEVICE, non_blocking=True)
            graphs[1].edge_index.to('cpu')

            BATCH_SIZE = images.shape[0]

            optimiser.zero_grad()

            ##################################################################

            # Ground Truth
            references = {}
            for b in range(BATCH_SIZE):
                captions = []
                for j in range(5):
                    caption = targets[b * 5 + j]
                    ref_caps = caption_array_to_string(convert(caption), 
                                                       is_scst=True)
                    captions.append(ref_caps)
                references[str(b)] = [cap for cap in captions]
            

            # Baseline
            model.eval()
            baselines = {}
            spatial_graphs = graphs[0].to_data_list()
            semantic_graphs = graphs[1].to_data_list()

            for b in range(BATCH_SIZE):
                # convert to single graph batch for GAT
                graphs_ = [Batch.from_data_list([spatial_graphs[b]]), Batch.from_data_list([semantic_graphs[b]])]
                hyp = caption_array_to_string(model.caption_image(graphs_, vocab, method='greedy'),
                                              is_scst=True)
                baselines[str(b)] = [hyp]
            model.train()



            # Sample decoding
            logits = model(graphs, targets[0:-1:5,:-1], lengths)
            probabilities = F.softmax(logits, dim=2)
            mask = torch.zeros_like(probabilities)
            B, T, V = probabilities.shape
            sampled = torch.zeros((B, T), dtype=torch.long)
            sampled_entropy = torch.zeros((B, T)).to(const.DEVICE)
            samples = {}
            for b in range(B):
                lens = lengths[b] - 1
                for t in range(lens):
                    sampled[b][t] = torch.multinomial(probabilities[b][t].view(-1), 1).item()
                    # sampled_entropy[b][t] = 1 - torch.log(probabilities[b][t][sampled[b][t]])
                    mask[b][t][sampled[b][t]] = 1                    

                caption = sampled[b]
                sampled_cap = caption_array_to_string(convert(caption), 
                                                      is_scst=True) 
                samples[str(b)] = [sampled_cap]            

            probabilities = probabilities - mask

            cider = Cider()
            cider_ = Cider()

            baseline_rewards = cider.compute_score(references, baselines)[1]
            reward = cider_.compute_score(references, samples)[1]

            log_prob = torch.sum(torch.sum(probabilities, dim=-1), dim=-1).to(const.DEVICE)

            baseline_rewards = torch.tensor(baseline_rewards).to(const.DEVICE)
            reward = torch.tensor(reward).to(const.DEVICE)

            loss = -torch.sum(log_prob * (reward - baseline_rewards))

            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        print(f"Loss for epoch {epoch}/{epoch_count} | {running_loss/len(train_data_loader)}")
    
    return model, epoch, None
            

@torch.no_grad()
def evaluate(model: nn.Module, 
             data_loader: DataLoader,
             split: str="validation"):
    model.eval()
    
    losses = []
    enumerator = tqdm(enumerate(data_loader))
    for idx, data in enumerator:
        images = data[0].to(const.DEVICE, non_blocking=True)
        
        targets = data[1].to(const.DEVICE, non_blocking=True)
        targets = targets[0:-1:5,:]
        
        lengths = data[2].to(const.DEVICE, non_blocking=True)
        lengths = lengths[0:-1:5]

        # Remove excess padding
        targets = targets[:,:max(lengths)] 
        
        if const.IS_GRAPH_MODEL:
            graphs = data[3]
            graphs[0].cuda()
            graphs[1].cuda()
            logits = model(graphs, targets, lengths)
            loss = F.cross_entropy(logits.permute(0, 2, 1), 
                                   targets[:,1:], 
                                   ignore_index=data_loader.dataset.vocab.stoi["<PAD>"])
        else: 
            logits = model(images, targets[:,:-1])
            loss = F.cross_entropy(logits.permute(0, 2, 1), 
                                   targets, 
                                   ignore_index=data_loader.dataset.vocab.stoi["<PAD>"])
        
        losses.append(loss.item())
    
    avg_loss = sum(losses)/len(losses)
    print(f"{split} loss: {avg_loss}")
    model.train()
    return avg_loss

