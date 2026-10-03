"""Dataset registry and factory.

`build_dataset(cfg) -> Splits(train, val, test)`: three dataset objects
sharing one vocab. For butd, word frequencies span the entire caption corpus
(all splits), and restval is folded into train per BUTD convention (train
113,287 images, val 5,000, test 5,000). Each object carries `.vocab`,
`.max_len`, `.split`.

Adding a dataset = write a `_name(data, seed) -> Splits` builder and register
it in BUILDERS; nothing else changes.
"""
import json
import os
import re
from collections import namedtuple
from pathlib import Path

from ..vocab import Vocab
from .base import BaseDataset, collate
from .synthetic import SyntheticDataset, WORDS
from .butd import ButdDataset

Splits = namedtuple("Splits", "train val test")
SPLITS = Splits._fields

# Vocab rule: one-hot ids for lowercase-alphanumeric words; anything else <unk>s.
_ALNUM = re.compile(r"[a-z0-9]+")

# repo-local vocab cache (gitignored); written on first build, reused after
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"   # parents[0]=datasets, [1]=captioning, [2]=repo root


def build_dataset(cfg: dict) -> Splits:
    data, seed = cfg["data"], cfg["seed"]
    builder = BUILDERS.get(data["dataset"])
    if builder is None:
        raise ValueError(f"unknown dataset {data['dataset']!r}; "
                         f"choose from {sorted(BUILDERS)}")
    return builder(data, seed)


def _synthetic(data: dict, seed: int) -> Splits:
    vocab = Vocab.from_captions([WORDS], 0)
    return Splits(*(SyntheticDataset(vocab, data["num_samples"], data["num_regions"],
                                     data["max_cap_len"], sp, seed) for sp in SPLITS))


def _butd(data: dict, seed: int) -> Splits:
    if not data["annotations_path"]:
        raise ValueError("data.annotations_path (dataset_coco.json) is required for the butd "
                         "dataset; machine paths live in the config")
    if not data["features_dir"]:
        raise ValueError("data.features_dir (butd_att dir) is required for the butd "
                         "dataset; machine paths live in the config")

    with open(data["annotations_path"]) as fh:
        imgs = json.load(fh)["images"]

    index = {sp: [(img["cocoid"], [s["tokens"] for s in img["sentences"]])
                  for img in imgs if img["split"] == sp] for sp in SPLITS + ("restval",)}
    index["train"] += index.pop("restval")  # BUTD convention: fold restval into train

    # Build Vocab over the whole caption corpus (all splits, incl. restval).
    # Cache key = annotations identity (path + mtime/size, so edits invalidate)
    # + vocab_min_freq + the alnum filter; bump the literal if the rule changes.
    all_caps = [s["tokens"] for img in imgs for s in img["sentences"]]
    st = os.stat(data["annotations_path"])
    cache_key = (f"{data['annotations_path']}|{st.st_mtime_ns}|{st.st_size}"
                 f"|{data['vocab_min_freq']}|alnum")
    vocab = Vocab.from_captions(all_caps, data["vocab_min_freq"],
                                keep=lambda w: _ALNUM.fullmatch(w) is not None,
                                cache_key=cache_key, cache_dir=CACHE_DIR)

    return Splits(*(ButdDataset(index[sp], vocab, data["features_dir"],
                                data["max_cap_len"], sp, seed) for sp in SPLITS))


BUILDERS = {"synthetic": _synthetic, "butd": _butd}
