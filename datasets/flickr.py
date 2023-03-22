import json
import os

import pandas as pd
import spacy
import torch
import torchvision.transforms as transforms
from torch_geometric.data import Batch, Data
from PIL import Image
from torch.utils.data import Dataset
from constants import Constants as const
from datasets.vocabulary import Vocabulary
from torchvision.io.image import read_image
import torchvision.transforms.functional as F
from utils.data_cleaning import preprocess_captions
import torchvision

class Flickr8kDataset(Dataset):
    def __init__(self, 
                root_dir:str, 
                captions_file: str, 
                transform:transforms.Compose=None, 
                freq_threshold: int=5):
        
        
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.imgs = list(self.df["image"].unique())
        self.captions = self.df["caption"]
        
        self.grouped_captions = pd.read_csv(captions_file, sep=',', header=None, names=['image', 'caption'])
        self.grouped_captions = self.grouped_captions.groupby('image')['caption'].apply(list).reset_index()

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img_id = self.imgs[index]
        # img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        # print(f'debug __getitem__: {index} -> {img_id}')
        img = read_image(os.path.join(self.root_dir, img_id))

        if self.transform is not None:
            img = self.transform(img)
        
        captions = self.grouped_captions.loc[self.grouped_captions['image'] == img_id]['caption'].values[0]
        captions = preprocess_captions(captions)

        return img, captions


    def get_grouped_captions(self, image_id: str):
        return self.grouped_captions.loc[self.grouped_captions['image'] == image_id]['caption'].item()


class Flickr8kDatasetWithSpatialGraphs(Flickr8kDataset):
    def __init__(self, 
                root_dir:str, 
                captions_file: str, 
                transform:transforms.Compose=None, 
                freq_threshold: int=5):
        
        super().__init__(root_dir, captions_file, transform, freq_threshold)
        self.graph_dir = const.PRECOMPUTED_SPATIAL_GRAPHS
        self.graphs = torch.load(self.graph_dir)
        for graph in self.graphs:
            self.graphs[graph].detach()
            self.graphs[graph].cpu()


    def __getitem__(self, index):
        img_id = self.imgs[index]
        img, captions = super().__getitem__(index)
        graph = self.graphs[img_id]
        # print(f'debug: Giving graph for {index} -> {img_id}')
        return img, captions, graph


class Flickr8kBatcher:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        self.vocab = Vocabulary(5)
        self.vocab.build_vocabulary()


    def __call__(self, data):
        def sorter(batch_element):
            length = len(batch_element[1][0].split(' '))
            return length

        data.sort(key=sorter, reverse=True)

        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Convert captions into numericalized captions.
        captions = [caption[0] for caption in captions]

        numericalized_captions = []
        for caption in captions:
            numericalized_caption = [self.vocab.stoi["<SOS>"]]
            numericalized_caption += self.vocab.numericalize(caption)
            numericalized_caption += [self.vocab.stoi["<EOS>"]]
            tensorised = torch.tensor(numericalized_caption)
            numericalized_captions.append(tensorised)

        lengths = [len(cap) for cap in numericalized_captions]

        captions_tensor = torch.zeros(len(numericalized_captions), max(lengths)).long()

        for i, cap in enumerate(numericalized_captions):
            end = lengths[i]
            captions_tensor[i, :end] = cap[:end]  
        return images, captions_tensor, torch.tensor(lengths, dtype=torch.int64)


class Flickr8kGraphsBatcher(Flickr8kBatcher):
    def __init__(self, pad_idx):
        super().__init__(pad_idx)


    def __call__(self, data):
        def sorter(batch_element):
            # Protect against hyphenated words (spacy tokeniser splits them)
            batch_element[1][0] = batch_element[1][0].replace('-', ' - ')
            return len(batch_element[1][0].split(' '))

        data.sort(key=sorter, reverse=True)

        images, captions, graphs = zip(*data)
        zipped = list(zip(images, captions))
        images, captions_tensor, lengths = super().__call__(zipped)

        graphs = Batch.from_data_list(list(graphs))

        return images, captions_tensor, lengths, graphs