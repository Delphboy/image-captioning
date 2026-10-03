"""Word-level vocabulary from Karpathy caption tokens.

Special tokens occupy ids 0-3: `<pad> <bos> <eos> <unk>`. Words are sorted by
frequency (desc, alphabetical tie-break) and filtered by a min frequency.

from_captions can cache the result: pass `cache_key` + `cache_dir` and a hit
loads `<cache_dir>/vocab_<sha1(cache_key)>.json` instead of recounting, a miss
computes and writes it. The caller owns the key — it must encode whatever
identifies the corpus/filter (path + mtime/size + min_freq...).
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)


class Vocab:
    def __init__(self, words: list[str]):
        """words: unique, already freq-sorted; specials prepended at 0-3."""
        self.itos = list(SPECIALS) + list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def from_captions(cls,
                      captions: list[list[str]],
                      min_freq: int = 0,
                      keep=None,
                      cache_key: str | None = None,
                      cache_dir: str | Path | None = None) -> "Vocab":
        """Words appearing at least min_freq times; keep (optional callable)
        further restricts which words may enter the vocab. With cache_key and
        cache_dir set, reuse the cached vocab when present, else compute +
        write it."""
        path = None
        if cache_key is not None and cache_dir is not None:
            path = Path(cache_dir) / f"vocab_{hashlib.sha1(cache_key.encode()).hexdigest()[:12]}.json"
            if path.is_file():
                try:
                    return cls.from_dict(json.loads(path.read_text()))
                except (ValueError, OSError):
                    pass  # corrupt/unreadable cache -> recompute below
        counts = Counter(w for cap in captions for w in cap)
        words = sorted((w for w, c in counts.items()
                        if c >= min_freq and (keep is None or keep(w))),
                       key=lambda w: (-counts[w], w))
        vocab = cls(words)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(vocab.to_dict()))
        return vocab

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, word: str) -> bool:
        return word in self.stoi

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(w, unk) for w in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def to_dict(self) -> dict:
        return {"words": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        words = d["words"]
        if words[:4] == list(SPECIALS):
            words = words[4:]
        return cls(words)
