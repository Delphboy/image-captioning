"""Smoke: tiny transformer on synthetic data trains; shapes + loss trend right."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from captioning.checkpoint import save_checkpoint
from captioning.config import load_config
from captioning.datasets import build_dataset, collate
from captioning.models import build as build_model
from captioning.trainer import Pretrainer


def make_cfg(tmp_path, **train_kwargs):
    p = Path(tmp_path) / "cfg.yaml"
    p.write_text(yaml.safe_dump({
        "model": {"name": "transformer", "d_model": 32, "n_heads": 4,
                  "n_layers": 2, "d_ff": 64, "dropout": 0.0},
        "data": {"dataset": "synthetic", "num_samples": 64, "max_cap_len": 16},
        "pretrain": {"batch_size": 8, "lr": 2e-3, "warmup_steps": 10,
                  "max_steps": 20, "log_every": 1, **train_kwargs},
    }))
    return load_config(p)


def test_logits_shape(tmp_path):
    cfg = make_cfg(tmp_path)
    ds = build_dataset(cfg)
    model = build_model(cfg, ds.train.vocab)
    batch = collate([ds.train[i] for i in range(4)])
    logits = model(batch)
    B, T = batch.cap_ids.shape
    assert logits.shape == (B, T, len(ds.train.vocab))
    assert logits[:, :-1].shape == (B, T - 1, len(ds.train.vocab))


def test_lr_annealing():
    from captioning.trainer import lr_at
    spe = 100  # steps per epoch
    # warmup phase: 1/4 of base at step 25; epoch 0 -> frac 0 -> no decay
    assert lr_at(25, 0.001, 100, 0, 0.8, 3, spe) == pytest.approx(0.001 * 0.25)
    # epast warmup; epoch 3 -> one decay step: *0.8
    assert lr_at(300, 0.001, 100, 0, 0.8, 3, spe) == pytest.approx(0.001 * 0.8)
    # epoch 10 -> frac (10-0)//3 = 3 -> *0.8**3
    assert lr_at(1000, 0.001, 100, 0, 0.8, 3, spe) == pytest.approx(0.001 * 0.8 ** 3)
    # delayed start: epoch 10 with start 4 -> frac (10-4)//3 = 2
    assert lr_at(1000, 0.001, 100, 4, 0.8, 3, spe) == pytest.approx(0.001 * 0.8 ** 2)
    # rate=0 disables annealing entirely
    assert lr_at(1000, 0.001, 100, 0, 0.0, 3, spe) == pytest.approx(0.001)


def test_train_loss_decreases(tmp_path):
    cfg = make_cfg(tmp_path)
    ds = build_dataset(cfg)
    model = build_model(cfg, ds.train.vocab)
    step, loss, history = Pretrainer(cfg, model, ds.train).run()
    assert step == cfg["pretrain"]["max_steps"]
    assert len(history) == 20  # log_every=1 -> one entry per step
    assert history[0][1] > history[-1][1], "loss must decrease over 20 steps"


def test_unknown_optim_and_loss_rejected(tmp_path):
    import pytest
    ds = build_dataset(make_cfg(tmp_path))
    model = build_model(make_cfg(tmp_path), ds.train.vocab)
    with pytest.raises(ValueError, match="optim"):
        Pretrainer(make_cfg(tmp_path, optim="rmsprop"), model, ds.train)
    with pytest.raises(ValueError, match="loss"):
        Pretrainer(make_cfg(tmp_path, loss="mse"), model, ds.train)