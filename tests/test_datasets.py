"""Datasets: synthetic contract, collate, BUTD loader (needs local data)."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from captioning.config import load_config
from captioning.datasets import BUILDERS, build_dataset, collate

COCO_JSON = Path("/home/henry/Datasets/coco/dataset_coco.json")
needs_coco = pytest.mark.skipif(not COCO_JSON.exists(),
                                reason="BUTD data not present on this machine")


def make_cfg(tmp_path, **data_kwargs):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"data": {"dataset": "synthetic", **data_kwargs}}))
    return load_config(p)


def test_registry_names():
    assert set(BUILDERS) == {"synthetic", "butd"}


def test_unknown_dataset_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataset(make_cfg(tmp_path, dataset="nope"))


def test_synthetic_contract(tmp_path):
    ds = build_dataset(make_cfg(tmp_path, num_samples=8, num_regions=36, max_cap_len=21))
    assert (ds.train.split, ds.val.split, ds.test.split) == ("train", "val", "test")
    for d in ds:                       # all three splits share the synthetic contract
        assert len(d) == 8
        assert d.max_len == 21
        assert d.vocab is not None
        assert len(d.vocab) > 4
    feat, ids, ln = ds.train[0]
    assert feat.shape == (36, 2048)
    assert feat.dtype == np.float32
    assert ids.dtype == np.int64
    assert ids[0] == 1 and ids[-1] == 2  # <bos>, <eos>
    assert ln == len(ids) <= 21
    assert 0 <= ids.min() and ids.max() < len(ds.train.vocab)


def test_synthetic_deterministic_with_seed():
    from captioning.config import merge_config
    d1 = build_dataset(merge_config({"data": {"dataset": "synthetic", "num_samples": 4}, "seed": 7}))
    d2 = build_dataset(merge_config({"data": {"dataset": "synthetic", "num_samples": 4}, "seed": 7}))
    assert np.array_equal(d1.train[0][0], d2.train[0][0])
    assert np.array_equal(d1.train[0][1], d2.train[0][1])
    assert np.array_equal(d1.test[0][0], d2.test[0][0])


def test_collate_pads_to_batch_max(tmp_path):
    from captioning.config import merge_config
    ds = build_dataset(merge_config({"data": {"dataset": "synthetic", "num_samples": 5, "max_cap_len": 16}}))
    items = [ds.train[i] for i in range(4)]
    batch = collate(items)
    T = max(it[2] for it in items)
    assert batch.feats.shape == (4, 36, 2048)
    assert batch.cap_ids.shape == batch.cap_mask.shape == (4, T)
    assert batch.feat_mask.shape == (4, 36)
    assert int(batch.feat_mask.numpy().sum()) == 4 * 36  # synthetic feats all real
    for b, (_, ids, ln) in enumerate(items):
        assert int(batch.cap_mask[b, :ln].numpy().sum()) == ln
        assert int(batch.cap_mask[b, ln:].numpy().sum()) == 0
        assert list(batch.cap_ids[b, :ln].numpy()) == list(ids)


def test_collate_pads_variable_regions(tmp_path):
    from captioning.config import merge_config
    ds8 = build_dataset(merge_config({"data": {"dataset": "synthetic", "num_samples": 2, "num_regions": 8}}))
    ds12 = build_dataset(merge_config({"data": {"dataset": "synthetic", "num_samples": 2, "num_regions": 12}}))
    batch = collate([ds8.train[0], ds12.train[0], ds8.train[1]])
    assert batch.feats.shape == (3, 12, 2048)   # padded to batch max N
    m = batch.feat_mask.numpy()
    assert m.tolist() == [[1] * 8 + [0] * 4,      # 8 real, 4 pad
                          [1] * 12,               # all real
                          [1] * 8 + [0] * 4]


@needs_coco
def test_butd_build_sample(tmp_path):
    cfg = make_cfg(tmp_path, dataset="butd", max_cap_len=21, vocab_min_freq=25,
                   annotations_path="/home/henry/Datasets/coco/dataset_coco.json",
                   features_dir="/home/henry/Datasets/coco/butd_att")
    ds = build_dataset(cfg)
    assert ds.val.split == "val"
    assert len(ds.val) == 5000
    assert 4 < len(ds.val.vocab) < 6000  # >=25 alnum words in the whole corpus: ~4800
    feat, ids, ln = ds.val[0]
    cocoid = ds.val.index[0][0]          # no region cap: N must match the raw npz
    assert feat.shape == np.load(f"/home/henry/Datasets/coco/butd_att/{cocoid}.npz")["feat"].shape
    assert feat.shape[1] == 2048
    assert feat.dtype == np.float32
    assert ids[0] == 1 and ids[-1] == 2
    assert ln <= 21


@needs_coco
def test_butd_restval_folded_into_train(tmp_path):
    cfg = make_cfg(tmp_path, dataset="butd",
                   annotations_path="/home/henry/Datasets/coco/dataset_coco.json",
                   features_dir="/home/henry/Datasets/coco/butd_att")
    ds = build_dataset(cfg)
    assert len(ds.train) == 82783 + 30504   # train + restval
    assert len(ds.test) == 5000
    assert ds.train.vocab is ds.val.vocab is ds.test.vocab  # one shared vocab


@needs_coco
def test_butd_split_filtering(tmp_path):
    with open(COCO_JSON) as fh:
        imgs = json.load(fh)["images"]
    val_ids = {i["cocoid"] for i in imgs if i["split"] == "val"}
    train_ids = {i["cocoid"] for i in imgs if i["split"] == "train"}
    restval_ids = {i["cocoid"] for i in imgs if i["split"] == "restval"}

    cfg = make_cfg(tmp_path, dataset="butd", vocab_min_freq=200,
                   annotations_path="/home/henry/Datasets/coco/dataset_coco.json",
                   features_dir="/home/henry/Datasets/coco/butd_att")
    ds = build_dataset(cfg)
    assert {c for c, _ in ds.train.index} == train_ids | restval_ids  # restval folded in
    assert not ({c for c, _ in ds.train.index} & val_ids)             # no val image leaks into train
    assert {c for c, _ in ds.test.index} == {i["cocoid"] for i in imgs if i["split"] == "test"}