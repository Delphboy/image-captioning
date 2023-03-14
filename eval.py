from constants import Constants as const

import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.flickr import Flickr8kVocabulary
from metrics.caption_metrics import bleu_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# TODO: This should probably go in utils.py
def caption_array_to_string(array: list[str]) -> str:
    caption = ""
    for i in range(1, len(array)):
        item = array[i]
        if item == "<EOS>": break

        # The captions.txt has a space before fullstops
        if item != '.':
            caption += f"{item} "
        else:
            caption += "."

    return caption


def evaluate_graph_caption_model(model: nn.Module,
                           loader: DataLoader,
                           dataset: Dataset):
    references = []
    hypotheses = []

    print("Generating Captions for Test Set")
    for idx, (imgs, captions, lengths, graphs) in tqdm(enumerate(loader), total=len(loader), leave=False):
        imgs = imgs.to(const.DEVICE)
        graphs.cuda()

        index = list(loader.dataset.indices)[idx]
        img_id = dataset.imgs[index]
        reference_captions = dataset.get_grouped_captions(img_id)

        prediction = model.caption_image_precomputed(graphs, dataset.vocab)
        candidate_caption = caption_array_to_string(prediction)
        
        hypotheses.append(candidate_caption)
        references.append(reference_captions)
        

    print("Calculating BLEU Score")
    print(bleu_score(references, hypotheses))



def evaluate_caption_model(model: nn.Module,
                           loader: DataLoader,
                           dataset: Dataset):
    references = []
    hypotheses = []

    print("Generating Captions for Test Set")
    for idx, (imgs, _, _) in tqdm(enumerate(loader), total=len(loader), leave=False):
        # print(f"Processing {idx+1}/{len(loader)}")
        imgs = imgs.to(const.DEVICE)

        index = list(loader.dataset.indices)[idx]
        img_id = dataset.imgs[index]
        reference_captions = dataset.get_grouped_captions(img_id)

        prediction = model.caption_image(imgs, dataset.vocab)
        candidate_caption = caption_array_to_string(prediction)
        
        hypotheses.append(candidate_caption)
        references.append(reference_captions)
        

    print("Calculating BLEU Score")
    print(bleu_score(references, hypotheses))
