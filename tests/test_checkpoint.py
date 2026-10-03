"""Checkpoint round-trip: save, load into a fresh model -> forward identical,
step + vocab restored; shape-mismatched keys skipped, not fatal."""
from pathlib import Path

import numpy as np
import yaml

from captioning.checkpoint import load_checkpoint, save_checkpoint
from captioning.config import load_config
from captioning.datasets import build_dataset, collate
from captioning.models import build as build_model
from captioning.models.transformer import Transformer
from captioning.trainer import Pretrainer


def make_cfg(tmp_path, **model_kwargs):
    p = Path(tmp_path) / "cfg.yaml"
    p.write_text(yaml.safe_dump({
        "model": {"name": "transformer", "d_model": 32, "n_heads": 4,
                  "n_layers": 2, "d_ff": 64, "dropout": 0.0, **model_kwargs},
        "data": {"dataset": "synthetic", "num_samples": 64, "max_cap_len": 16},
        "pretrain": {"batch_size": 8, "lr": 2e-3, "warmup_steps": 10,
                  "max_steps": 6, "log_every": 1},
    }))
    return load_config(p)


def test_experiment_prefixes_checkpoint_name(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["experiment"] = "expA"
    ds = build_dataset(cfg)
    model = build_model(cfg, ds.train.vocab)
    path = save_checkpoint(cfg, model, ds.train.vocab, 3, Path(tmp_path) / "ck")
    assert path.name == "expA_step_3.npz"
    path = save_checkpoint(cfg, model, ds.train.vocab, 3, Path(tmp_path) / "ck",
                           name="step_3-best.npz")
    assert path.name == "expA_step_3-best.npz"
    step, _, _ = load_checkpoint(path, model)  # prefixed file loads like any other
    assert step == 3


def test_no_experiment_is_plain_name(tmp_path):
    cfg = make_cfg(tmp_path)
    ds = build_dataset(cfg)
    model = build_model(cfg, ds.train.vocab)
    path = save_checkpoint(cfg, model, ds.train.vocab, 3, Path(tmp_path) / "ck")
    assert path.name == "step_3.npz"


def test_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path)
    ds = build_dataset(cfg)
    model = build_model(cfg, ds.train.vocab)
    step, _, _ = Pretrainer(cfg, model, ds.train).run()

    batch = collate([ds.train[i] for i in range(4)])
    before = model(batch).numpy()

    other = build_model(cfg, ds.train.vocab)  # fresh init == "perturbed" params
    path = save_checkpoint(cfg, model, ds.train.vocab, step, Path(tmp_path) / "ck")
    step2, cfg2, vocab2 = load_checkpoint(path, other)
    after = other(batch).numpy()

    np.testing.assert_allclose(before, after, atol=1e-6)
    assert step2 == step
    assert cfg2 == cfg
    assert vocab2.to_dict() == ds.train.vocab.to_dict()


def test_shape_mismatch_skipped(tmp_path):
    cfg = make_cfg(tmp_path)
    ds = build_dataset(cfg)
    model = build_model(cfg, ds.train.vocab)
    step, _, _ = Pretrainer(cfg, model, ds.train).run()
    path = save_checkpoint(cfg, model, ds.train.vocab, step, Path(tmp_path) / "ck")

    wrong = Transformer(len(ds.train.vocab), d_model=16, n_heads=2, n_layers=2,
                        d_ff=32, dropout=0.0, max_len=16)  # d_model mismatch
    step2, _, _ = load_checkpoint(path, wrong)  # must warn+skip, not raise
    assert step2 == step