import json
import os
from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset


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

        with open("/homes/hps01/image-captioning/datasets/cocotalk.json", "r") as f:
            self.coco_talk = json.load(f)

        self.vocab = list(self.coco_talk['ix_to_word'].values())
        
        # invert a dictionary
        self.word_to_ix = {v: int(k) for k, v in self.coco_talk["ix_to_word"].items()}


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
        with open("/homes/hps01/image-captioning/datasets/cocotalk.json", "r") as f:
            self.coco = json.load(f)

        # invert a dictionary
        self.word_to_ix = {v: k for k, v in self.coco["ix_to_word"].items()}


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
            numericalized_caption += [self.coco_word_to_ix("<END>")]
            tensorised = torch.tensor(numericalized_caption)
            numericalized_captions.append(tensorised)

        lengths = [len(cap) for cap in numericalized_captions]
        targets = torch.zeros(len(numericalized_captions), max(lengths)).long()

        for i, cap in enumerate(numericalized_captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]  
        return images, targets, torch.tensor(lengths, dtype=torch.int64)