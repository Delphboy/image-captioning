from typing import Tuple
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from datasets.flickr import Flickr8kDataset, MyCollate


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
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return train_loader, test_loader, dataset

