#!/usr/bin/env python3
"""main.py: --config PATH [--ckpt NPZ] [--max-steps N]

Pretrain on the train split (val loop runs per eval.every_steps), save a final
checkpoint, then generate captions for the test split — printing each image id
and predicted caption — and score them with BLEU/METEOR/ROUGE/CIDEr.
"""
import argparse
from pathlib import Path

import numpy as np
from tinygrad import Tensor

from captioning.checkpoint import load_checkpoint, save_checkpoint
from captioning.config import load_config, resolve_device
from captioning.datasets import build_dataset
from captioning.eval import evaluate, generate
from captioning.models import build as build_model
from captioning.trainer import Pretrainer


def main():
    ap = argparse.ArgumentParser(description="Train an image captioning model on tinygrad")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--ckpt", help="npz checkpoint to resume from")
    ap.add_argument("--max-steps", type=int, help="override pretrain.max_steps cap")
    args = ap.parse_args()

    # Load and validate configu
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["pretrain"]["max_steps"] = args.max_steps
    resolve_device(cfg)

    # Set seeds
    np.random.seed(cfg["seed"])
    Tensor.manual_seed(cfg["seed"])

    # dataset creation -> Splits(train, val, test)
    train, val, test = build_dataset(cfg)
    print(f"{len(train)} train / {len(val)} val / {len(test)} test images")
    print(f"vocab: {len(train.vocab)} tokens")

    # pretrain
    model = build_model(cfg, train.vocab)
    trainer = Pretrainer(cfg, model, train, val)
    start_step = 0
    if args.ckpt:
        start_step, _, ckpt_vocab = load_checkpoint(Path(args.ckpt), model)
        if len(ckpt_vocab) != len(train.vocab):
            print(f"WARNING: checkpoint vocab {len(ckpt_vocab)} != dataset vocab {len(train.vocab)}")
        print(f"resumed from {args.ckpt} at step {start_step}")
    step, loss, _ = trainer.run(start_step=start_step)
    ckpt_dir = Path(cfg["pretrain"]["ckpt_dir"])
    path = save_checkpoint(cfg, model, train.vocab, step, ckpt_dir)
    print(f"done: step {step} loss {loss:.4f} -> {path}")

    # test-set evaluation: image id + predicted caption each, then metrics
    print(f"\n--- test captions ({len(test)} images) ---")
    for image_id, caption in generate(model, test, cfg["eval"]):
        print(f"{image_id}: {caption}")
    metrics = evaluate(model, test, cfg["eval"])
    print("\n--- metrics ---")
    for name, score in sorted(metrics.items()):
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    main()