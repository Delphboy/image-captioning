"""Eval: beam/greedy decode sanity on a stub dataset, and real coco-caption
metric scoring (BLEU/METEOR/ROUGE/CIDEr) through the submodule shim."""
import sys
from pathlib import Path

import numpy as np
import pytest

from captioning.checkpoint import load_checkpoint
from captioning.datasets import build_dataset
from captioning.eval import _EvalData, _coco_evalcap, evaluate, generate
from captioning.models import build as build_model
from captioning.vocab import Vocab

COCO_CAPTION = Path(__file__).resolve().parents[1] / "coco-caption"
needs_submodule = pytest.mark.skipif(not (COCO_CAPTION / "pycocoevalcap").exists(),
                                     reason="coco-caption submodule not checked out")


class StubDS:
    """Dataset-shaped object with an index (cocoid -> refs) and npz-like feats."""

    def __init__(self):
        rng = np.random.RandomState(0)
        self.vocab = Vocab.from_captions([["a", "dog", "cat", "runs", "fast"]], min_freq=0)
        self.max_len = 8
        self.index = [(1000 + i, [["a", "dog", "runs"], ["a", "cat", "runs"]])
                      for i in range(4)]
        self._feats = [rng.rand(rng.randint(4, 10), 2048).astype(np.float32) for _ in range(4)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self._feats[i], np.array([1, 2], dtype=np.int64), 2  # captions unused


def make_model():
    from captioning.config import merge_config
    from tinygrad import Tensor
    cfg = merge_config({"model": {"name": "transformer", "d_model": 32, "n_heads": 4,
                                  "n_layers": 1, "d_ff": 64, "dropout": 0.0},
                        "data": {"dataset": "synthetic"}})
    Tensor.manual_seed(0)
    return build_model(cfg, StubDS().vocab)


def test_generate_greedy_and_beam(tmp_path):
    ds = StubDS()
    model = make_model()
    for w in (1, 3):
        caps = generate(model, ds, {"beam_width": w, "batch_size": 2, "max_samples": 3})
        assert len(caps) == 3                      # max_samples respected
        assert [c for c, _ in caps] == [c for c, _ in ds.index[:3]]  # cocoids in order
        for _, text in caps:
            toks = text.split()
            assert len(toks) <= ds.max_len
            assert all(t in ds.vocab for t in toks), f"token outside vocab: {toks}"


def test_eval_runs_on_stub(tmp_path):
    ds = StubDS()
    model = make_model()
    out = evaluate(model, ds, {"beam_width": 1, "batch_size": 2, "max_samples": 2})
    assert {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"} <= set(out)
    assert all(v >= 0.0 for v in out.values())


@needs_submodule
def test_coco_metrics_identical_references():
    """References identical to results must score perfectly."""
    gts = {1: [{"caption": "a dog runs fast"}, {"caption": "a dog running fast"}],
           2: [{"caption": "a cat sits on a mat"}, {"caption": "a cat sits on the mat"}]}
    res = {1: [{"caption": "a dog runs fast"}], 2: [{"caption": "a cat sits on a mat"}]}
    COCOEvalCap = _coco_evalcap()
    ev = COCOEvalCap(_EvalData(gts), _EvalData(res))
    ev.evaluate(use_spice=False)
    assert ev.eval["Bleu_1"] == pytest.approx(1.0)
    assert set(ev.eval) == {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"}


def test_eval_saves_checkpoints_and_best_cider(tmp_path, monkeypatch):
    """Every eval saves a resumable checkpoint; -best only on CIDEr improvement."""
    import captioning.trainer as tr
    from captioning.config import merge_config
    from captioning.trainer import Pretrainer

    scores = {"Bleu_1": 0.8, "Bleu_2": 0.7, "Bleu_3": 0.6, "Bleu_4": 0.5,
              "METEOR": 0.3, "ROUGE_L": 0.4, "CIDEr": 1.0}
    evals = iter([scores, {**scores, "CIDEr": 0.5}])          # best at step 1, worse at step 2
    monkeypatch.setattr(tr, "evaluate", lambda model, val_ds, cfg_eval: next(evals))

    cfg = merge_config({"model": {"name": "transformer", "d_model": 32, "n_heads": 4,
                                  "n_layers": 1, "d_ff": 64, "dropout": 0.0},
                        "data": {"dataset": "synthetic"},
                        "pretrain": {"batch_size": 4, "lr": 1e-3, "warmup_steps": 1,
                                      "max_steps": 2, "log_every": 1,
                                      "ckpt_dir": str(tmp_path / "ck")},
                        "eval": {"every_steps": 1, "beam_width": 1,
                                  "batch_size": 2, "max_samples": 2}})
    ds = StubDS()
    model = make_model()
    Pretrainer(cfg, model, ds, val_ds=ds).run()

    ck = Path(cfg["pretrain"]["ckpt_dir"])
    assert sorted(p.name for p in (ck / "eval").glob("*.npz")) == ["step_1.npz", "step_2.npz"]
    assert [p.name for p in sorted(ck.glob("*-best.npz"))] == ["step_1-best.npz"]  # 0.5 < 1.0: no overwrite
    step, cfg2, vocab2 = load_checkpoint(ck / "step_1-best.npz", make_model())
    assert step == 1 and cfg2 == cfg and len(vocab2) == len(ds.vocab)  # resumable