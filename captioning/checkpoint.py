"""np.savez checkpoints: model params (get_state_dict) + step + vocab + cfg.

Filenames carry an optional `<experiment>_` prefix read from cfg["experiment"]
(no prefix when unset). Meta values are stored as JSON strings (keys prefixed
`_`, impossible for param names), so loading needs no pickle.
"""
import json
from pathlib import Path

import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict, load_state_dict

from .vocab import Vocab


def save_checkpoint(cfg: dict, model, vocab: Vocab, step: int, ckpt_dir: Path, name: str | None = None) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp = cfg.get("experiment")
    prefix = f"{exp}_" if exp else ""
    path = ckpt_dir / f"{prefix}{name if name else f'step_{step}.npz'}"
    arrays = {k: v.numpy() for k, v in get_state_dict(model).items()}
    arrays["_step"] = np.array(step)
    arrays["_cfg"] = np.array(json.dumps(cfg))
    arrays["_vocab"] = np.array(json.dumps(vocab.to_dict()))
    np.savez(path, **arrays)
    print(f"saved checkpoint {path}")
    return path


def load_checkpoint(path: Path, model) -> tuple[int, dict, Vocab]:
    """Restore model params (warn + skip keys missing or shape-mismatched on
    the model, since tinygrad's load_state_dict raises on mismatch). Returns
    (step, cfg, vocab)."""
    with np.load(path, allow_pickle=False) as z:
        step, cfg, vocab = int(z["_step"]), json.loads(str(z["_cfg"])), \
            Vocab.from_dict(json.loads(str(z["_vocab"])))
        saved = {k: Tensor(z[k]) for k in z.files if not k.startswith("_")}
    model_sd = get_state_dict(model)
    usable = {k: v for k, v in saved.items()
              if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    for k, v in saved.items():
        if k not in usable:
            print(f"WARNING: skipping {k} (shape {tuple(v.shape)} missing/differs "
                  f"from model {tuple(model_sd.get(k).shape) if k in model_sd else None})")
    load_state_dict(model, usable, strict=False)
    return step, cfg, vocab
