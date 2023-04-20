import json
import os
from typing import Any, Callable, List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torchvision.datasets import VisionDataset
from torchvision.io.image import read_image
from torchvision.io import ImageReadMode

from constants import Constants as const
from datasets.vocabulary import Vocabulary


class CocoCaptionsDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        with open(const.TALK_FILE, "r") as f:
            self.coco_talk = json.load(f)

        # invert a dictionary
        self.word_to_ix = {v: int(k) for k, v in self.coco_talk.items()}
        self.vocab = Vocabulary(5)
        self.vocab.build_vocabulary(list(self.coco_talk.values()))
        print()


    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")


    def _load_target(self, id: int) -> List[Any]:
        return [ann["caption"] for ann in self.coco.loadAnns(self.coco.getAnnIds(id))]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def __len__(self) -> int:
        return len(self.ids)


class CocoBatcher:
    def __init__(self):
        with open(const.TALK_FILE, "r") as f:
            self.coco = json.load(f)

        # invert a dictionary
        self.word_to_ix = {v: k for k, v in self.coco.items()}


    def coco_ix_to_word(self, ix):
        return self.coco.ix_to_word[ix]


    def coco_word_to_ix(self, word):
        res = self.word_to_ix[word]
        return int(res)


    def captions_to_numbers(self, caption):
        numericalized_caption = []
        sanitised_caption = caption.lower().split(' ')
        for word in sanitised_caption:
            if word in self.word_to_ix:
                numericalized_caption.append(self.coco_word_to_ix(word))
            else:
                numericalized_caption.append(self.coco_word_to_ix("<UNK>"))
        return numericalized_caption


    def __call__(self, data):
        def sorter(batch_element):
            return len(batch_element[1][0].split(' '))

        data.sort(key=sorter, reverse=True)

        images, captions = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = torch.stack(images, 0)

        # Merge captions (from tuple of 1D tensor to 2D tensor).
        captions = [caption[0] for caption in captions]

        numericalized_captions = []
        for caption in captions:
            numericalized_caption = [self.coco_word_to_ix("<SOS>")]
            numericalized_caption += self.captions_to_numbers(caption)
            numericalized_caption += [self.coco_word_to_ix("<EOS>")]
            tensorised = torch.tensor(numericalized_caption)
            numericalized_captions.append(tensorised)

        lengths = [len(cap) for cap in numericalized_captions]
        targets = torch.zeros(len(numericalized_captions), max(lengths)).long()

        for i, cap in enumerate(numericalized_captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]  
        return images, targets, torch.tensor(lengths, dtype=torch.int64)


class CocoGraphsBatcher(CocoBatcher):
    def __init__(self):
        super().__init__()


    def __call__(self, data):
        def sorter(batch_element):
            return len(batch_element[1][0].split(' '))

        data.sort(key=sorter, reverse=True)
        images, captions, graphs = zip(*data)

        zipped = list(zip(images, captions))
        images, captions_tensor, lengths = super().__call__(zipped)
        graphs = Batch.from_data_list(list(graphs))
        
        return images, captions_tensor, lengths, graphs


class CocoKarpathy(Dataset):
    def __init__(self, 
                 root_dir: str, # /import/gameai-01/eey362/datasets/coco/images
                 captions_file: str, # splits/dataset_coco.json
                 transform:transforms.Compose=None, 
                 freq_threshold: int=5,
                 split: str='train'):
        
        self.root_dir = root_dir
        self.captions_file = captions_file
        self.transform = transform
        self.freq_threshold = freq_threshold
        assert split in ['train', 'val', 'test'], f'Split must be train, val or test. Received: {split}'
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
                self.data[image['cocoid']] = {
                    'dir': image['filepath'],
                    'filename': image['filename'],
                    'sentences': [sentence['raw'] for sentence in image['sentences']]
                }
            
            captions += [sentence['raw'] for sentence in image['sentences']]

        self.ids = list(self.data.keys())

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(captions)

        if const.IS_GRAPH_MODEL:
            self.spatial_graphs = torch.load(const.PRECOMPUTED_SPATIAL_GRAPHS[self.split]) if const.PRECOMPUTED_SPATIAL_GRAPHS else None
            self.semantic_graphs = torch.load(const.PRECOMPUTED_SEMANTIC_GRAPHS[self.split]) if const.PRECOMPUTED_SEMANTIC_GRAPHS else None



    def __getitem__(self, index):
        data_id = self.ids[index]
        data = self.data[data_id]
        image = read_image(os.path.join(self.root_dir, data['dir'], data['filename']), ImageReadMode.RGB)
        captions = data['sentences']

        if self.transform is not None:
            image = self.transform(image)

        if const.IS_GRAPH_MODEL:
            spatial_graph = self.spatial_graphs[data_id]
            # spatial_graph.edge_index = spatial_graph.edge_index.to(torch.float32)

            # semantic_graph = self.semantic_graphs[data_id]
            return image, captions, spatial_graph

        return image, captions


    def __len__(self):
        return len(self.ids)

