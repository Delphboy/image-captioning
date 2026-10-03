"""Synthetic dataset: random features + captions, for tests and smoke runs."""
import numpy as np

from ..vocab import Vocab
from .base import BaseDataset

WORDS = ["dog", "cat", "bird", "car", "man", "woman", "child", "street", "sky",
         "grass", "red", "blue", "white", "running", "sitting", "eating",
         "small", "large", "near", "on"]


class SyntheticDataset(BaseDataset):
    def __init__(self, vocab: Vocab, num_samples: int = 100, num_regions: int = 36,
                 max_cap_len: int = 21, split: str = "train", seed: int = 42):
        self.vocab = vocab
        self.max_len = max_cap_len
        self.split = split
        rng = np.random.RandomState(seed)
        self.feat = rng.rand(num_samples, num_regions, 2048).astype(np.float32)
        self.caps = [[WORDS[rng.randint(len(WORDS))] for _ in range(rng.randint(3, max_cap_len - 1))]
                     for _ in range(num_samples)]

    def __len__(self) -> int:
        return len(self.caps)

    def __getitem__(self, i: int):
        ids = [self.vocab.stoi["<bos>"]] + self.vocab.encode(self.caps[i]) \
            + [self.vocab.stoi["<eos>"]]
        ids = np.array(ids[: self.max_len], dtype=np.int64)
        return self.feat[i], ids, len(ids)