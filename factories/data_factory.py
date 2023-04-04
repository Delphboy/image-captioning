from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.coco import CocoBatcher, CocoCaptionsDataset, CocoKarpathy
from datasets.flickr import Flickr8kDataset, Flickr8kDatasetWithSpatialGraphs, CaptionBatcher, GraphsCaptionBatch
from constants import Constants as const

def _get_flickr8k_data() -> Tuple[DataLoader, DataLoader, Dataset, Dataset, int]:
    if const.IS_GRAPH_MODEL:
        dataset = Flickr8kDatasetWithSpatialGraphs(const.ROOT, 
                                                   const.ANNOTATIONS, 
                                                   transform=const.STANDARD_TRANSFORM)
        pad_idx = dataset.vocab.stoi["<PAD>"]
        batcher = GraphsCaptionBatch(pad_idx=pad_idx)
    else:
        dataset = Flickr8kDataset(const.ROOT, 
                                  const.ANNOTATIONS, 
                                  transform=const.STANDARD_TRANSFORM)
        pad_idx = dataset.vocab.stoi["<PAD>"]
        batcher = CaptionBatcher(pad_idx=pad_idx)
    
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


def _get_coco_data() -> Tuple[DataLoader, DataLoader, Dataset, Dataset, int]:
    # TODO: Change call to CocoCaptionsDataset to use os.path.join
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


def _get_coco_karpathy_data() -> Tuple[DataLoader, DataLoader, DataLoader, Dataset, Dataset, Dataset, int]:
    train_dataset = CocoKarpathy(root_dir=const.ROOT,
                                 captions_file=const.ANNOTATIONS,
                                 transform=const.STANDARD_TRANSFORM,
                                 split='train',
                                 freq_threshold=5)
    
    test_dataset = CocoKarpathy(root_dir=const.ROOT,
                                captions_file=const.ANNOTATIONS,
                                transform=const.STANDARD_TRANSFORM,
                                split='test',
                                freq_threshold=5)
    
    val_dataset = CocoKarpathy(root_dir=const.ROOT,
                               captions_file=const.ANNOTATIONS,
                               transform=const.STANDARD_TRANSFORM,
                               split='val',
                               freq_threshold=5)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=const.BATCH_SIZE,
        num_workers=const.NUM_WORKERS,
        shuffle=const.SHUFFLE,
        pin_memory=const.PIN_MEMORY,
        collate_fn=CaptionBatcher(train_dataset.vocab.stoi["<PAD>"]),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=const.PIN_MEMORY,
        collate_fn=CaptionBatcher(train_dataset.vocab.stoi["<PAD>"]),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=const.PIN_MEMORY,
        collate_fn=CaptionBatcher(train_dataset.vocab.stoi["<PAD>"]),
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_dataset.vocab.stoi["<PAD>"]


DATASETS = {
    "flickr8k": _get_flickr8k_data,
    "coco": _get_coco_data,
    "coco_karpathy": _get_coco_karpathy_data
}


def get_data(dataset_name: str):
    
    if dataset_name not in DATASETS:
        raise Exception(f"The dataset '{dataset_name}' is not supported by the factory. Supported models are {DATASETS.keys()}")

    return DATASETS[dataset_name]()
