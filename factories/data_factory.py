from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.coco import CocoBatcher, CocoCaptionsDataset
from datasets.flickr import Flickr8kBatcher, Flickr8kDataset


def get_flickr8k_data(
    root_folder,
    annotation_file,
    transform,
    train_ratio=0.8,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
) -> Tuple[DataLoader, DataLoader, Dataset]:
    dataset = Flickr8kDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                [train_size, test_size], 
                                                                 generator=torch.Generator().manual_seed(42))


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Flickr8kBatcher(pad_idx=pad_idx),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=Flickr8kBatcher(pad_idx=pad_idx),
    )

    return train_loader, test_loader, dataset


def get_coco_data(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
) -> Tuple[DataLoader, DataLoader, Dataset]:
    train_dataset = CocoCaptionsDataset(f"{root_folder}train2017", 
                                        f"{annotation_file}captions_train2017.json", 
                                        transform=transform)
    
    val_dataset = CocoCaptionsDataset(f"{root_folder}val2017", 
                                      f"{annotation_file}captions_val2017.json", 
                                      transform=transform)
       

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=CocoBatcher(),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=CocoBatcher(),
    )

    return train_loader, val_loader, train_dataset, val_dataset