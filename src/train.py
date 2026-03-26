from datasets import get_dataloader, get_dataset
from opts import parse_args

from tinygrad import Tensor


if __name__ == "__main__":
    args = parse_args()

    dataset = get_dataset(args)
    img_count = len(dataset)

    print(f"Total images in split={args.split}: {img_count}")

    dataloader = get_dataloader(args)
    first_batch = next(iter(dataloader))
    print(f"Batch image shape: {first_batch['img'].shape}")
    print(f"Batch captions: {len(first_batch['captions'])}")

    print((Tensor([1,2,3]) + Tensor([3,4,5])).tolist())
