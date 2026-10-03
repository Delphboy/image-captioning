"""Pretraining loop: forward -> loss -> backward -> optimizer step, with
linear warmup then constant LR, and periodic console logging. Agonstic to
dataset and model — it feeds each batch to `model(batch)` and applies the
loss/optimizer named in the config.

Optimizers and losses are registries keyed by config string, so a new one is
a one-line addition: `OPTIMS = {..., "rmsprop": nn.optim.RMSprop}`. Any loss
takes `(logits, batch)` and returns a scalar Tensor — the captioning CE
builds targets from `batch.cap_ids` (predict token t+1 from positions 0..t,
pad positions excluded via ignore_index=0).

tinygrad's "training mode" is the TRAINING context var (no model.training
attr in this pin): the whole loop runs inside Context(TRAINING=1), which both
enables dropout and satisfies the optimizer's TRAINING check.
"""
import math
import time
from pathlib import Path

import numpy as np
from tinygrad import nn, Tensor
from tinygrad.helpers import Context, TRAINING

from .checkpoint import save_checkpoint
from .datasets.base import collate
from .eval import evaluate


def lr_at(step: int, lr: float, warmup_steps: int, decay_start: int = 0,
          decay_rate: float = 0.8, decay_every: int = 3,
          steps_per_epoch: int = 1) -> float:
    """Linear warmup to lr, then step-anneal every decay_every epochs after
    decay_start (epoch = step // steps_per_epoch):
    lr *= decay_rate ** ((epoch - decay_start) // decay_every). rate=0 disables."""
    epoch = step // max(1, steps_per_epoch)
    lr = lr * min(1.0, step / max(1, warmup_steps))
    if decay_rate > 0 and decay_every > 0 and epoch >= decay_start:
        lr = lr * decay_rate ** ((epoch - decay_start) // decay_every)
    return lr


def ce_loss(logits: Tensor, batch) -> Tensor:
    """Captioning cross-entropy; pad positions masked via ignore_index=0."""
    return logits[:, :-1].sparse_categorical_crossentropy(batch.cap_ids[:, 1:], ignore_index=0)


LOSSES = {"ce": ce_loss}
OPTIMS = {"adam": nn.optim.Adam, "adamw": nn.optim.AdamW, "sgd": nn.optim.SGD, "muon": nn.optim.Muon}


class Pretrainer:
    def __init__(self,
                 cfg: dict,
                 model,
                 dataset,
                 val_ds=None):
        self.cfg, self.model, self.ds = cfg, model, dataset
        self.val_ds = val_ds
        np.random.seed(cfg["seed"])  # batch order (data itself is seeded at build)
        t = cfg["pretrain"]
        optim = OPTIMS.get(t["optim"])
        if optim is None:
            raise ValueError(f"unknown optim {t['optim']!r}; choose from {sorted(OPTIMS)}")
        if t["loss"] not in LOSSES:
            raise ValueError(f"unknown loss {t['loss']!r}; choose from {sorted(LOSSES)}")
        self.loss_fn = LOSSES[t["loss"]]
        self.opt = optim(nn.state.get_parameters(model), lr=t["lr"])
        self.best_cider = float("-inf")  # in-memory best; resets on resume
        self._best_path = None

    def run(self, start_step: int = 0) -> tuple[int, float, list[tuple[int, float]]]:
        """Train until max_steps (or epochs*batches), returning (step, last_loss, history)."""
        t = self.cfg["pretrain"]
        steps_per_epoch = math.ceil(len(self.ds) / t["batch_size"])
        total = t["max_steps"] if t["max_steps"] else t["epochs"] * steps_per_epoch
        remaining = max(1, total - start_step)

        def batches():
            while True:
                idx = np.random.permutation(len(self.ds))
                for i in range(0, len(idx), t["batch_size"]):
                    yield collate([self.ds[j] for j in idx[i:i + t["batch_size"]]])

        history, last_loss = [], float("nan")
        with Context(TRAINING=1):
            data = batches()
            t0 = time.perf_counter()
            for step in range(start_step + 1, start_step + remaining + 1):
                batch = next(data)
                logits = self.model(batch)
                loss = self.loss_fn(logits, batch)

                self.opt.zero_grad()
                loss.backward()

                lr = lr_at(step, t["lr"], t["warmup_steps"],
                           t["learning_rate_decay_start"], t["learning_rate_decay_rate"],
                           t["learning_rate_decay_every"], steps_per_epoch)
                self.opt.lr.assign(lr)
                self.opt.step()

                last_loss = float(loss.item())
                if step % t["log_every"] == 0:
                    dt = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    sps = t["log_every"] / dt if dt > 0 else 0.0
                    print(f"step {step:6d} loss {last_loss:8.4f} lr {lr:.2e} {sps:6.1f} steps/s",
                          flush=True)
                    history.append((step, last_loss))

                every = self.cfg["eval"]["every_steps"]
                if every and step % every == 0 and self.val_ds is not None:
                    metrics = evaluate(self.model, self.val_ds, self.cfg["eval"])
                    summary = " ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items()))
                    print(f"eval @ step {step}: {summary}", flush=True)
                    ckpt_dir = Path(self.cfg["pretrain"]["ckpt_dir"])
                    save_checkpoint(self.cfg, self.model, self.val_ds.vocab, step,
                                    ckpt_dir / "eval")  # resumable eval checkpoint
                    cider = metrics.get("CIDEr", float("-inf"))
                    if cider > self.best_cider:
                        self.best_cider = cider
                        if self._best_path is not None and self._best_path.exists():
                            self._best_path.unlink()
                        self._best_path = save_checkpoint(
                            self.cfg, 
                            self.model, 
                            self.val_ds.vocab, 
                            step, 
                            ckpt_dir, 
                            f"step_{step}-best.npz")

        return start_step + remaining, last_loss, history
