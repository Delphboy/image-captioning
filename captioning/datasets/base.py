"""Dataset base class and batch collate.

Dataset protocol: `__len__`, `__getitem__(i) -> (feat[np (N,2048)],
cap_ids[np (L,)], cap_len:int)` — N varies per image in butd adaptive mode;
after `build_dataset(cfg)` the object carries `.vocab`, `.max_len`, `.split`.
"""
import numpy as np
from tinygrad import Tensor
from collections import namedtuple


class BaseDataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int):
        raise NotImplementedError


# What a model receives per step; a model reads only the fields it needs, so
# datasets may add fields without touching models or the trainer.
Batch = namedtuple("Batch", "feats cap_ids cap_mask feat_mask")


def collate(items: list) -> Batch:
    """Pad region features to the batch's max N and captions to max T.

    Returns a Batch: feats (B,Nmax,2048), cap_ids (B,T) int, cap_mask (B,T)
    float, feat_mask (B,Nmax) bool; masks are 1/True for real tokens/regions,
    0/False for padding. feat_mask must be bool: SDPA turns bool masks into
    additive -inf on padded keys.
    """
    N = max(it[0].shape[0] for it in items)
    feats = np.zeros((len(items), N, items[0][0].shape[1]), dtype=np.float32)
    feat_mask = np.zeros((len(items), N), dtype=bool)
    T = max(it[2] for it in items)
    cap_ids = np.zeros((len(items), T), dtype=np.int64)
    cap_mask = np.zeros((len(items), T), dtype=np.float32)
    for b, (feat, ids, ln) in enumerate(items):
        feats[b, : feat.shape[0]] = feat
        feat_mask[b, : feat.shape[0]] = True
        if ln <= T:
            cap_ids[b, :ln] = ids[:T]
            cap_mask[b, :ln] = 1.0
    return Batch(Tensor(feats), Tensor(cap_ids), Tensor(cap_mask), Tensor(feat_mask))