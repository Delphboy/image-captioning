from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.coco import CocoBatcher, CocoCaptionsDataset
from datasets.flickr import Flickr8kDataset, Flickr8kDatasetWithSpatialGraphs, Flickr8kBatcher, Flickr8kGraphsBatcher
from constants import Constants as const

def _get_flickr8k_data() -> Tuple[DataLoader, DataLoader, Dataset]:
        
    if const.IS_GRAPH_MODEL:
        dataset = Flickr8kDatasetWithSpatialGraphs(const.ROOT, 
                                                   const.ANNOTATIONS, 
                                                   transform=const.STANDARD_TRANSFORM)
        pad_idx = dataset.vocab.stoi["<PAD>"]
        batcher = Flickr8kGraphsBatcher(pad_idx=pad_idx)
    else:
        dataset = Flickr8kDataset(const.ROOT, 
                                  const.ANNOTATIONS, 
                                  transform=const.STANDARD_TRANSFORM)
        pad_idx = dataset.vocab.stoi["<PAD>"]
        batcher = Flickr8kBatcher(pad_idx=pad_idx)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                [train_size, test_size], 
                                                                 generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=const.BATCH_SIZE,
        num_workers=const.NUM_WORKERS,
        shuffle=const.SHUFFLE,
        pin_memory=const.PIN_MEMORY,
        collate_fn=batcher
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=const.PIN_MEMORY,
        collate_fn=batcher,
    )

    return train_loader, test_loader, train_dataset, test_dataset, dataset.vocab.stoi["<PAD>"]


def _get_coco_data() -> Tuple[DataLoader, DataLoader, Dataset]:
    train_dataset = CocoCaptionsDataset(f"{const.ROOT}train2017", 
                                        f"{const.ANNOTATIONS}captions_train2017.json", 
                                        transform=const.STANDARD_TRANSFORM)
    
    val_dataset = CocoCaptionsDataset(f"{const.ROOT}val2017", 
                                      f"{const.ANNOTATIONS}captions_val2017.json", 
                                      transform=const.STANDARD_TRANSFORM)
       

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=const.BATCH_SIZE,
        num_workers=const.NUM_WORKERS,
        shuffle=const.SHUFFLE,
        pin_memory=const.PIN_MEMORY,
        collate_fn=CocoBatcher(),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=const.PIN_MEMORY,
        collate_fn=CocoBatcher(),
    )

    return train_loader, val_loader, train_dataset, val_dataset, train_dataset.word_to_ix['<PAD>']



DATASETS = {
    "flickr8k": _get_flickr8k_data,
    "coco_train": _get_coco_data
}

def get_data(dataset_name: str):
    
    if dataset_name not in DATASETS:
        raise Exception(f"The dataset '{dataset_name}' is not supported by the factory. Supported models are {DATASETS.keys()}")

    return DATASETS[dataset_name]()