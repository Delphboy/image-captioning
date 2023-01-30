from constants import Constants as const

import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.flickr import Flickr8kVocabulary
from metrics.caption_metrics import bleu_score, meteor_score
from torch.utils.data import DataLoader, Dataset

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


def evaluate_caption_model(model: nn.Module,
                           loader: DataLoader,
                           dataset: Dataset):
    ground_truths = []
    predictions = []

    for idx, (imgs, captions, lengths) in enumerate(loader):
        print(f"Processing {idx+1}/{len(loader)}")
        imgs = imgs.to(const.DEVICE)
        captions = captions.to(const.DEVICE)[0]

        
        predict = model.caption_image(imgs, dataset.vocab)
        
        # print(f"Predicted: {caption_array_to_string(predict)}")       
        ground_truth = [dataset.vocab.itos[i.item()] for i in captions]
        # print(f"Should be: {ground_truth}")
        
        predictions.append(caption_array_to_string(predict))
        ground_truths.append(caption_array_to_string(ground_truth))

    print(bleu_score(ground_truths, predictions))
