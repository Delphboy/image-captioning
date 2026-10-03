"""Validation: beam-search caption generation + coco-caption metrics.

`evaluate(model, val_ds, cfg_eval) -> {Bleu_1..4, METEOR, ROUGE_L, CIDEr}`
generates one caption per validation image (beam search; beam_width=1 is
greedy) and scores them against the image's reference captions using the
coco-caption submodule's COCOEvalCap (BLEU/METEOR/ROUGE/CIDEr; SPICE off).

Notes:
- Generation runs inside Context(TRAINING=0) so dropout is off even when
  called from inside the Pretrainer's TRAINING=1 loop.
- COCOEvalCap only touches `getImgIds()` + `imgToAnns`, so references and
  results are handed to it as plain dicts via _EvalData — the annotation
  objects never need the real pycocotools COCO machinery. pip pycocotools is
  imported first only to satisfy the evaluator's ecosystem assumption.
- coco-caption's directory name has a hyphen, so it is loaded via sys.path
  insertion; importing pycocotools (pip) first keeps the submodule's stale
  bundled copy from being picked up afterwards.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import Context, TRAINING

from .vocab import BOS, EOS
from .datasets.base import Batch

COCO_CAPTION_DIR = Path(__file__).resolve().parents[1] / "coco-caption"


# ---------------------------------------------------------------- generation

def beam_decode(model, feats: Tensor, feat_mask: Tensor, W: int, max_len: int,
                vocab_size: int, bos_id: int, eos_id: int) -> list[np.ndarray]:
    """Beam-search decode feats (B,N,2048) -> one caption id array per image.

    W=1 collapses to greedy. All beams run to max_len (no early stop); finished
    beams freeze their score and compete via a zero-gain <eos> continuation.
    The best-scoring beam per image is cut at its first <eos>.
    """
    B = feats.shape[0]
    feats_w = feats.repeat_interleave(W, dim=0)          # (B*W, N, 2048)
    mask_w = feat_mask.repeat_interleave(W, dim=0)
    seqs = np.full((B, W, 1), bos_id, dtype=np.int64)
    scores = np.zeros((B, W), dtype=np.float32)
    done = np.zeros((B, W), dtype=bool)

    for t in range(1, max_len):
        flat = seqs.reshape(B * W, t)
        logits = model(Batch(feats_w, Tensor(flat), Tensor(np.ones((B * W, t), dtype=bool)), mask_w))
        lp = logits[:, -1].numpy().reshape(B, W, vocab_size)          # (B,W,V)
        lp -= lp.max(-1, keepdims=True)
        lp -= np.log(np.exp(lp).sum(-1, keepdims=True))               # log-softmax

        ext = scores[:, :, None] + lp                                 # (B,W,V)
        ext = np.where(done[:, :, None], -np.inf, ext)                # freeze finished beams
        bw_done = np.where(done)
        ext[bw_done[0], bw_done[1], eos_id] = scores[bw_done]         # ...selectable via <eos>

        flat_ext = ext.reshape(B, W * vocab_size)
        if W > 1:
            top = np.argpartition(-flat_ext, W - 1, axis=1)[:, :W]
        else:
            top = flat_ext.argmax(axis=1, keepdims=True)
        top_scores = flat_ext[np.arange(B)[:, None], top]
        order = np.argsort(-top_scores, axis=1)                       # deterministic tie-break
        top = np.take_along_axis(top, order, axis=1)
        scores = np.take_along_axis(top_scores, order, axis=1)
        bw, v = np.divmod(top, vocab_size)                            # (B,W) beam + token
        seqs = np.concatenate(
            [seqs[np.arange(B)[:, None], bw], v[:, :, None]], axis=2)
        done = done[np.arange(B)[:, None], bw] | (v == eos_id)

    best = scores.argmax(axis=1)
    out = []
    for cap in seqs[np.arange(B), best]:
        eos_at = np.where(cap == eos_id)[0]
        out.append(cap[: int(eos_at[0])] if len(eos_at) else cap)
    return out


def generate(model, ds, cfg_eval: dict) -> list[tuple[int, str]]:
    """Greedy/beam-decode one caption per image in ds (respecting max_samples)."""
    if not hasattr(ds, "index"):
        raise ValueError("eval requires a dataset with reference captions (.index)")
    W = cfg_eval["beam_width"]
    bs = cfg_eval["batch_size"]
    idx = list(range(len(ds)))
    if cfg_eval.get("max_samples"):
        idx = idx[: cfg_eval["max_samples"]]
    captions = []
    with Context(TRAINING=0):  # dropout off during generation
        for i in range(0, len(idx), bs):
            items = [ds[j] for j in idx[i:i + bs]]
            N = max(it[0].shape[0] for it in items)
            feats = np.zeros((len(items), N, items[0][0].shape[1]), dtype=np.float32)
            feat_mask = np.zeros((len(items), N), dtype=bool)
            for b, it in enumerate(items):
                feats[b, : it[0].shape[0]] = it[0]
                feat_mask[b, : it[0].shape[0]] = True
            caps = beam_decode(model, Tensor(feats), Tensor(feat_mask), W,
                               ds.max_len, len(ds.vocab),
                               ds.vocab.stoi[BOS], ds.vocab.stoi[EOS])
            for b, cap in enumerate(caps):
                toks = ds.vocab.decode(cap.tolist())
                if toks and toks[0] == BOS:
                    toks = toks[1:]
                if EOS in toks:
                    toks = toks[: toks.index(EOS)]
                captions.append((ds.index[idx[i + b]][0], " ".join(toks)))
    return captions


# ---------------------------------------------------------------- scoring

class _EvalData:
    """Minimal getImgIds/imgToAnns stand-in for COCOEvalCap."""

    def __init__(self, anns: dict):
        self.imgToAnns = anns
        self._ids = list(anns)

    def getImgIds(self) -> list:
        return self._ids


def _coco_evalcap():
    import pycocotools  # noqa: F401  (cache pip's copy before path insertion)
    if str(COCO_CAPTION_DIR) not in sys.path:
        sys.path.insert(0, str(COCO_CAPTION_DIR))
    return importlib.import_module("pycocoevalcap.eval").COCOEvalCap


def evaluate(model, val_ds, cfg_eval: dict) -> dict[str, float]:
    """Generate captions for val_ds and score them with coco-caption metrics."""
    res = generate(model, val_ds, cfg_eval)
    index = dict(val_ds.index)
    gts = {c: [{"caption": " ".join(t)} for t in index[c]] for c, _ in res}
    results = {c: [{"caption": text}] for c, text in res}

    COCOEvalCap = _coco_evalcap()
    evaluator = COCOEvalCap(_EvalData(gts), _EvalData(results))
    evaluator.evaluate(use_spice=False)  # BLEU/METEOR/ROUGE/CIDEr only
    return {k: float(v) for k, v in evaluator.eval.items()}