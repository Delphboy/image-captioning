"""Config loading: YAML file -> dict, merged over fixed defaults.

Strict: unknown keys raise ValueError, a non-mapping top level refuses to
load, and numeric strings (YAML 1.1 resolves `2e-3` to a string) are coerced
on merge. ckpt_dir resolves relative to the config file's directory, so
checkpoint output lands predictably regardless of the process CWD. Device
resolution sets TINYGRAD_DEVICE when the config names an explicit device, so
the same code runs on any backend (auto = tinygrad picks).
"""
import os
from pathlib import Path

import yaml

DEFAULTS = {
    "device": "auto",
    "seed": 1,
    "experiment": None,  # checkpoint filenames carry an <experiment>_ prefix; None = plain
    "model": {"name": "transformer", "d_model": 256, "n_heads": 8, "n_layers": 4,
              "d_ff": 1024, "dropout": 0.1},
    "data": {"dataset": "synthetic",
             "annotations_path": None,  # butd: dataset_coco.json path; machine paths live in configs, not code
             "features_dir": None,      # butd: butd_att/ dir of per-image npz features
             "num_regions": 36,  # synthetic-only; butd adaptive mode has no region cap
             "max_cap_len": 21,
             "vocab_min_freq": 5,
             "num_samples": 100},  # synthetic-only; ignored by butd
    "pretrain": {"epochs": 1, "batch_size": 32, "lr": 1e-4, "warmup_steps": 1000,
                 "learning_rate_decay_start": 0, "learning_rate_decay_rate": 0.8,
                 "learning_rate_decay_every": 3,
                 "max_steps": None, "log_every": 10, "ckpt_dir": "checkpoints",
                 "optim": "adamw", "loss": "ce"},
    "eval": {"every_steps": None,  # val loop runs every N pretrain steps; null = off
              "beam_width": 1,      # 1 = greedy decode
              "batch_size": 64,
              "max_samples": None},  # cap on val images scored; None = full split
}


def _coerce(v):
    """Coerce numeric-looking strings (PyYAML/YAML 1.1 resolves `2e-3` to the
    string "2e-3"); non-numeric strings pass through unchanged."""
    if isinstance(v, str):
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
    return v


def _merge(dst, src):
    if not isinstance(src, dict):
        raise ValueError(f"config section must be a mapping, got {type(src).__name__}")
    extra = sorted(set(src) - set(dst))
    if extra:
        raise ValueError(f"unknown config keys: {extra}")
    for key, default in dst.items():
        if isinstance(default, dict):
            dst[key] = _merge(dict(default), src[key] if key in src else {})
        elif key in src and isinstance(default, (int, float)):
            dst[key] = _coerce(src[key])  # numeric-defaulted leaves only; string knobs stay strings
        elif key in src:
            dst[key] = src[key]
    return dst


def merge_config(user: dict) -> dict:
    """Merge a user config dict over DEFAULTS; unknown keys raise ValueError."""
    return _merge({k: (dict(v) if isinstance(v, dict) else v)
                   for k, v in DEFAULTS.items()}, user)


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and merge it over the defaults.

    ckpt_dir resolves relative to the config file's directory, so checkpoint
    output is independent of the process CWD (L0-02).
    """
    p = Path(path)
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}  # empty file == defaults
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: config top level must be a mapping, got {type(cfg).__name__}")
    cfg = merge_config(cfg)
    cfg["pretrain"]["ckpt_dir"] = str(p.resolve().parent / cfg["pretrain"]["ckpt_dir"])
    return cfg


def resolve_device(cfg: dict) -> str | None:
    """Set TINYGRAD_DEVICE when cfg['device'] is explicit; return it, else None."""
    dev = cfg["device"]
    if dev != "auto":
        os.environ["TINYGRAD_DEVICE"] = dev
        return dev
    return None
