from constants import Constants as const
from utils.helper_functions import caption_array_to_string

import torch.nn as nn
from torch.utils.data import DataLoader
from metrics.caption_metrics import bleu_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEBUG=False 

def evaluate_graph_caption_model(model: nn.Module,
                           loader: DataLoader,
                           dataset: Dataset):
    references = []
    hypotheses = []
    img_idx = 0
    model.eval()
    print("Generating Captions for Test Set")
    for data in tqdm(iter(dataset), total=len(dataset), leave=False):
        imgs = data[0].to(const.DEVICE)
        imgs = imgs.unsqueeze(0)
        reference_captions = data[1]
        graphs = data[2]
        graphs.cuda()

        prediction = model.caption_image_precomputed(graphs, dataset.dataset.vocab)
        candidate_caption = caption_array_to_string(prediction)
        if DEBUG: print(f"{dataset.dataset.imgs[dataset.indices[img_idx]]}: {candidate_caption}")
        
        hypotheses.append(candidate_caption)
        references.append(reference_captions)
        img_idx += 1
        

    print("Calculating BLEU Score")
    print(bleu_score(references, hypotheses))



def evaluate_caption_model(model: nn.Module,
                           loader: DataLoader,
                           dataset: Dataset):
    references = []
    hypotheses = []
    
    img_idx = 0
    model.eval()
    print("Generating Captions for Test Set")
    for data in tqdm(iter(dataset), total=len(dataset), leave=False):
        imgs = data[0].to(const.DEVICE)
        imgs = imgs.unsqueeze(0)
        reference_captions = data[1]

        prediction = model.caption_image(imgs, dataset.dataset.vocab)
        candidate_caption = caption_array_to_string(prediction)
        if DEBUG: print(f"{dataset.dataset.imgs[dataset.indices[img_idx]]}: {candidate_caption}")
        
        hypotheses.append(candidate_caption)
        references.append(reference_captions)
        img_idx += 1
        

    print("Calculating BLEU Score")
    print(bleu_score(references, hypotheses))
