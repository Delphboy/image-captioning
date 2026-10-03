"""Config: strict merge (unknown keys raise), numeric-string coercion,
config-relative ckpt_dir, device resolution."""
import os

import pytest
import yaml

from captioning.config import load_config, merge_config, resolve_device


def write(tmp_path, **user):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(user))
    return p


def test_defaults_filled_when_keys_missing(tmp_path):
    cfg = load_config(write(tmp_path, model={"d_model": 128}))
    assert cfg["device"] == "auto"
    assert cfg["seed"] == 1
    assert cfg["experiment"] is None
    assert cfg["model"]["d_model"] == 128
    assert cfg["model"]["n_heads"] == 8
    assert cfg["data"]["dataset"] == "synthetic"
    assert cfg["data"]["max_cap_len"] == 21
    assert cfg["pretrain"]["batch_size"] == 32
    assert cfg["pretrain"]["max_steps"] is None
    assert cfg["pretrain"]["optim"] == "adamw"
    assert cfg["pretrain"]["loss"] == "ce"
    assert cfg["pretrain"]["learning_rate_decay_start"] == 0
    assert cfg["pretrain"]["learning_rate_decay_rate"] == 0.8
    assert cfg["pretrain"]["learning_rate_decay_every"] == 3


def test_load_empty_tmp_yaml_returns_defaults(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("")
    assert load_config(p)["model"]["d_model"] == 256


def test_unknown_keys_raise(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        load_config(write(tmp_path, bogus=1))
    with pytest.raises(ValueError, match="wat"):
        load_config(write(tmp_path, model={"wat": 2, "d_model": 64}))


def test_malformed_top_level_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_numeric_strings_coerced(tmp_path):
    cfg = load_config(write(tmp_path, pretrain={"lr": "2e-3", "batch_size": "8"}))
    assert cfg["pretrain"]["lr"] == 0.002
    assert cfg["pretrain"]["batch_size"] == 8


def test_features_dir_default_none(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg["data"]["features_dir"] is None
    assert cfg["data"]["annotations_path"] is None


def test_ckpt_dir_config_relative(tmp_path):
    assert load_config(
        write(tmp_path, pretrain={"ckpt_dir": "out"}))["pretrain"]["ckpt_dir"] == str(tmp_path / "out")


def test_deep_override_nested(tmp_path):
    cfg = load_config(write(tmp_path, pretrain={"lr": 1e-3}, data={"num_regions": 20}))
    assert cfg["pretrain"]["lr"] == 1e-3
    assert cfg["pretrain"]["epochs"] == 1
    assert cfg["data"]["num_regions"] == 20


def test_unknown_data_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="split"):
        load_config(write(tmp_path, data={"split": "val"}))  # splits come from build_dataset(), not config


def test_device_auto_resolves_to_none(monkeypatch):
    monkeypatch.delenv("TINYGRAD_DEVICE", raising=False)
    assert resolve_device(merge_config({})) is None
    assert "TINYGRAD_DEVICE" not in os.environ


def test_device_explicit_sets_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TINYGRAD_DEVICE", raising=False)
    cfg = load_config(write(tmp_path, device="cuda"))
    assert resolve_device(cfg) == "cuda"
    assert os.environ["TINYGRAD_DEVICE"] == "cuda"