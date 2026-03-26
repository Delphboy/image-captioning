from typing import Any, Dict

import json
import os

from PIL import Image


class Coco:
    def __init__(self, opts: Any):
        self.dataset = opts.dataset
        self.img_root = opts.img_root
        self.split = opts.split

        required_dirs = ["train2014", "val2014", "test2014"]
        for dir_name in required_dirs:
            if dir_name not in os.listdir(self.img_root):
                raise AssertionError(f"DIRECTORY NOT FOUND: COCO img root does not contain '{dir_name}'")

        with open(opts.dataset_json, "r") as f:
            data = json.load(f)

        images = data["images"]

        if self.split == "test":
            self.images = [image for image in images if image["split"] == "test"]
        elif self.split == "val":
            self.images = [image for image in images if image["split"] == "val"]
        elif self.split == "train":
            self.images = [image for image in images if image["split"] not in ["val", "test"]]
        else:
            self.images = images


    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index out of range: {idx}")

        return {
            "img": self._get_img(idx),
            "raw_captions": self._get_raw_captions(idx),
        }

    def _get_img(self, idx: int) -> Image.Image:
        image_data = self.images[idx]
        image_path = os.path.join(self.img_root, image_data["filepath"], image_data["filename"])
        return Image.open(image_path).convert("RGB")

    def _get_raw_captions(self, idx: int) -> list[str]:
        image_data = self.images[idx]
        return [sentence["raw"] for sentence in image_data["sentences"]]
