import json
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torchvision.io import ImageReadMode
from torchvision.io.image import read_image

from constants import Constants as const
from datasets.vocabulary import Vocabulary
from utils.data_cleaning import preprocess_captions


class Flickr8kDataset(Dataset):
    def __init__(self, 
                root_dir:str, 
                captions_file: str, 
                transform:transforms.Compose=None, 
                freq_threshold: int=5,
                split='train') -> None:
        
        assert split in ['train', 'val', 'test'], f'Split must be train, val or test. Received: {split}'

        self.root_dir = root_dir
        self.captions_file = captions_file
        self.transform = transform
        self.freq_threshold = freq_threshold
        self.split = split

        # captions_file is a json file. Load it into a dictionary
        with open(self.captions_file, 'r') as f:
            self.captions_file_data = json.load(f)

        self.data = {}
        captions = []

        for image in self.captions_file_data['images']:
            if image['split'] == 'restval':
                image['split'] = 'train'

            if image['split'] == self.split:
                self.data[image['imgid']] = {
                    # 'dir': image['filepath'],
                    'filename': image['filename'],
                    'sentences': [sentence['raw'] for sentence in image['sentences']]
                }
            
            captions += [sentence['raw'] for sentence in image['sentences']]

        self.ids = list(self.data.keys())

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(captions)



    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        index = list(self.data.keys())[index]
        img_id = self.data[index]['filename']
        # print(f'debug __getitem__: {index} -> {img_id}')
        img = read_image(os.path.join(self.root_dir, img_id), ImageReadMode.RGB)

        if self.transform is not None:
            img = self.transform(img)
        
        # captions = self.grouped_captions.loc[self.grouped_captions['image'] == img_id]['caption'].values[0]
        captions = self.data[index]['sentences']
        captions = preprocess_captions(captions)

        return img, captions



class Flickr8kDatasetWithSpatialGraphs(Flickr8kDataset):
    def __init__(self, 
                root_dir:str, 
                captions_file: str, 
                transform:transforms.Compose=None, 
                freq_threshold: int=5,
                split='train') -> None:
        
        super().__init__(root_dir, captions_file, transform, freq_threshold, split)
        
        self.graphs = torch.load(const.PRECOMPUTED_SPATIAL_GRAPHS[self.split]) if const.PRECOMPUTED_SPATIAL_GRAPHS else None
        for graph in self.graphs:
            self.graphs[graph].detach()
            self.graphs[graph].cpu()


    def __getitem__(self, index):
        new_index = list(self.data.keys())[index]
        img_id = self.data[new_index]['filename']
        img, captions = super().__getitem__(index)
        graph = self.graphs[new_index]
        graph.edge_index = graph.edge_index.to(torch.float32)
        # print(f'debug: Giving graph for {index} -> {img_id}')
        return img, captions, graph


class CaptionBatcher:
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


class GraphsCaptionBatcher(CaptionBatcher):
    def __init__(self, pad_idx):
        super().__init__(pad_idx)


    def __call__(self, data):
        def sorter(batch_element):
            return len(batch_element[1][0].split(' '))

        data.sort(key=sorter, reverse=True)

        images, captions, graphs = zip(*data)
        zipped = list(zip(images, captions))
        images, captions_tensor, lengths = super().__call__(zipped)

        graphs = Batch.from_data_list(list(graphs)) #FIXME: I break

        return images, captions_tensor, lengths, graphs