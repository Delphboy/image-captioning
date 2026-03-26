from typing import Any, Dict, Iterator, List

import random


class DatasetDataLoader:
    def __init__(self, dataset: Any, batch_size: int = 8, image_size: int = 224, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if len(batch_indices) < self.batch_size:
                break
            yield self._build_batch(batch_indices)

    def _build_batch(self, batch_indices: List[int]) -> Dict[str, Any]:
        from tinygrad.tensor import Tensor

        imgs: List[list[list[list[float]]]] = []
        captions: List[List[str]] = []

        for idx in batch_indices:
            sample = self.dataset[idx]
            imgs.append(self._pil_to_chw(sample["img"]))
            captions.append(sample["captions"])

        return {"img": Tensor(imgs), "captions": captions}

    def _pil_to_chw(self, img) -> list[list[list[float]]]:
        resized = img.resize((self.image_size, self.image_size))
        pixels = resized.load()

        red: list[list[float]] = []
        green: list[list[float]] = []
        blue: list[list[float]] = []

        for y in range(self.image_size):
            row_r: list[float] = []
            row_g: list[float] = []
            row_b: list[float] = []
            for x in range(self.image_size):
                r, g, b = pixels[x, y]
                row_r.append(r / 255.0)
                row_g.append(g / 255.0)
                row_b.append(b / 255.0)
            red.append(row_r)
            green.append(row_g)
            blue.append(row_b)

        return [red, green, blue]
