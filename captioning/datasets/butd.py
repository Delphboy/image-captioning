"""BUTD COCO loader: Karpathy json + region features from per-image npz files.

features_dir points at the butd_att/ directory of <cocoid>.npz feature files.
Adaptive mode: feature counts vary per image (10-100 regions), returned
as-is with no num_regions truncation. One of the image's 5 captions is
sampled per fetch; collate pads regions to the batch max.
"""
import os
import random

import numpy as np

from ..vocab import Vocab, BOS, EOS
from .base import BaseDataset


class ButdDataset(BaseDataset):
    def __init__(self,
                 index: list,
                 vocab: Vocab,
                 features_dir: str,
                 max_cap_len: int = 21,
                 split: str = "train",
                 seed: int = 1):
        # index: list of (cocoid, [token_list, ...]) for images in this split
        self.index = index
        self.vocab = vocab
        self.max_len = max_cap_len
        self.split = split
        self.features_dir = features_dir
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        cocoid, sentences = self.index[i]
        feat = np.load(os.path.join(self.features_dir, f"{cocoid}.npz"))["feat"]
        tokens = self.rng.choice(sentences)
        ids = [self.vocab.stoi[BOS]] + self.vocab.encode(tokens) + [self.vocab.stoi[EOS]]
        ids = np.array(ids[: self.max_len], dtype=np.int64)
        return feat.astype(np.float32), ids, len(ids)
