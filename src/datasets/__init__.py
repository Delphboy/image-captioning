from .coco import Coco
from .dataloader import DatasetDataLoader
from .flickr8k import Flickr8K


def get_dataset(opts):
    if opts.dataset == "flickr8k":
        return Flickr8K(opts)
    if opts.dataset == "coco":
        return Coco(opts)
    raise ValueError(f"Unsupported dataset: {opts.dataset}")


def get_dataloader(opts, shuffle: bool = False):
    dataset = get_dataset(opts)
    return DatasetDataLoader(
        dataset=dataset,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        shuffle=shuffle,
    )
