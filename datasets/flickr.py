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


class Flickr8kVocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text): return [tok.text.lower() for tok in const.SPACY_ENG.tokenizer(text)]

    def build_vocabulary(self, sentence_list=None):
        # if flickr8k.json exists, load it into self.itos and self.stoi
        if os.path.exists('datasets/flickrtalk.json'):
            with open('datasets/flickrtalk.json', 'r') as f:
                self.itos = json.load(f)
                self.stoi = {v: int(k) for k, v in self.itos.items()}
            return

        frequencies = {}
        idx = 4 # idx 0, 1, 2, 3 are already taken (PAD, SOS, EOS, UNK)
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = int(idx)
                    self.itos[idx] = word
                    idx += 1
        
        # write self.itos to a json file in teh datasets folder
        with open('datasets/flickrtalk.json', 'w') as f:
            json.dump(self.itos, f)

    # TODO: Make sure that the length of vocab can be expressed as a power of 2: https://twitter.com/karpathy/status/1621578354024677377?s=20

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


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

        self.vocab = Flickr8kVocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        captions = self.grouped_captions.loc[self.grouped_captions['image'] == img_id]['caption'].values[0]

        return img, captions


    def get_grouped_captions(self, image_id: str):
        return self.grouped_captions.loc[self.grouped_captions['image'] == image_id]['caption'].item()


class Flickr8kDatasetWithSpatialGraphs(Flickr8kDataset):
    def __init__(self, 
                root_dir:str, 
                captions_file: str, 
                transform:transforms.Compose=None, 
                freq_threshold: int=5,
                graph_dir: str=None):
        
        super().__init__(root_dir, captions_file, transform, freq_threshold)
        # TODO: Add check to compute graphs if the graph_dir is None
        self.graph_dir = graph_dir
        self.graphs = torch.load(self.graph_dir)
        for graph in self.graphs:
            self.graphs[graph].detach()
            self.graphs[graph].cpu()


    def __getitem__(self, index):
        img, captions = super().__getitem__(index)
        graph = self.graphs[index]
        return img, captions, graph


class Flickr8kBatcher:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        self.vocab = Flickr8kVocabulary(5)
        self.vocab.build_vocabulary()


    def __call__(self, data):
        def sorter(batch_element):
            # Protect against hyphenated words (spacy tokeniser splits them)
            batch_element[1][0] = batch_element[1][0].replace('-', ' - ')
            return len(batch_element[1][0].split(' '))

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

        targets = torch.zeros(len(numericalized_captions), max(lengths)).long()

        for i, cap in enumerate(numericalized_captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]  
        return images, targets, torch.tensor(lengths, dtype=torch.int64)


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
        images, targets, lengths = super().__call__(zipped)

        graphs = Batch.from_data_list(list(graphs))

        return images, targets, lengths, graphs