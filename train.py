import os
import sys
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
from pycocoevalcap.eval import Rouge, Bleu, Meteor
from eval import evaluate_caption_model

# sys.path.append("pyciderevalcap")
# from cider.pyciderevalcap.ciderD.ciderD import CiderD

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
            # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 0.1)
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
                        scheduler: optim.lr_scheduler._LRScheduler,
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
            lengths = lengths[0:-1:5] # Only want lengths of captions being predicted
            
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
                references[b] = [cap for cap in captions]
            

            baselines = {}
            spatial_graphs = graphs[0].to_data_list()
            semantic_graphs = graphs[1].to_data_list()

            logits = model(graphs, targets[0:-1:5,:-1], lengths)
            probs = F.softmax(logits, dim=2)
            pred_idx = torch.argmax(probs, dim=-1)
            
            samples = {}
            sampled_probs = torch.zeros((BATCH_SIZE, pred_idx.shape[1])).to(const.DEVICE)
            
            for b in range(BATCH_SIZE):
                # Samples
                for t in range(pred_idx.shape[1]):
                    sampled_probs[b][t] = probs[b][t][pred_idx[b][t]]
                samples[b] = [caption_array_to_string(convert(pred_idx[b]), is_scst=True)]
                
                # Baselines
                _graphs = [Batch.from_data_list([spatial_graphs[b]]), Batch.from_data_list([semantic_graphs[b]])] # convert to single graph batch for GAT
                hyp = caption_array_to_string(model.caption_image(_graphs, vocab, method='greedy'),
                                              is_scst=True)
                baselines[b] = [hyp]


            log_probabilities = torch.log(sampled_probs)
            log_probabilities = log_probabilities.mean(dim=-1)

            cider = Rouge() # TODO: Switch to CiderD using the new library
            cider_ = Rouge()
            # c = CiderD(df='coco-val')
            # _r = c.compute_score(references, samples)

            reward = cider_.compute_score(references, samples)[1]
            reward = torch.tensor(reward).to(const.DEVICE)
            
            baseline_rewards = cider.compute_score(references, baselines)[1]
            baseline_rewards = torch.tensor(baseline_rewards).to(const.DEVICE)

            loss = -log_probabilities * (reward - baseline_rewards)
            loss = loss.mean()

            ##################################################################

            # print(f"Reference: {[references]}")
            # print(f"Test time: {baselines}")
            # print(f"Train time: {samples}")
            # print()

            ##################################################################
            
            print(f"Loss for batch {idx}: {loss.item()}")

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 0.1)
            optimiser.step()
            running_loss += loss.item()

        print(f"Loss for epoch {epoch}/{epoch_count} | {running_loss/len(train_data_loader)}")
        scheduler.step()
        val_loss = evaluate(model, val_data_loader)
        val_loss_vals.append([const.EPOCHS + epoch, val_loss])
        train_loss = evaluate(model, train_data_loader)
        train_loss_vals.append([const.EPOCHS + epoch, train_loss])       
        
        # Test the model on the validation set with CIDEr loss
        if epoch == 1 or epoch % 5 == 0:            
            global_results, _ = evaluate_caption_model(model, val_data_loader.dataset)
            val_performance_vals.append([const.EPOCHS + epoch, global_results])
            model.train()
    
    return model, const.EPOCHS + epoch, None
            

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

