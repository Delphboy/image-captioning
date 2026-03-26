import argparse
import os

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image captioning options")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["flickr8k", "coco"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="Path to Karpathy-style dataset JSON",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        required=True,
        help="Path to image root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Dataset split to load",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for dataloader",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Square image size for resizing",
    )

    return parser


def validate_dataset_args(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.dataset_json):
        raise AssertionError(f"FILE NOT FOUND: dataset json file not found: {args.dataset_json}")

    if not os.path.isdir(args.img_root):
        raise AssertionError(f"DIRECTORY NOT FOUND: image root not found: {args.img_root}")

    if args.batch_size <= 0:
        raise AssertionError("INVALID ARG: --batch_size must be > 0")

    if args.image_size <= 0:
        raise AssertionError("INVALID ARG: --image_size must be > 0")


def parse_args() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    validate_dataset_args(args)
    return args
