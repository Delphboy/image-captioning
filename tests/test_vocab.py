"""Vocab: special ids 0-3, encode/decode round-trip, min-freq filtering."""
from captioning.vocab import Vocab

CAPS = [["a", "man", "with", "a", "hat"],
        ["a", "woman", "with", "a", "dog"],
        ["a", "dog", "on", "a", "street"]]


def test_special_token_ids():
    v = Vocab.from_captions(CAPS, min_freq=1)
    assert v.stoi["<pad>"] == 0
    assert v.stoi["<bos>"] == 1
    assert v.stoi["<eos>"] == 2
    assert v.stoi["<unk>"] == 3


def test_round_trip():
    v = Vocab.from_captions(CAPS, min_freq=0)
    for cap in CAPS:
        assert v.decode(v.encode(cap)) == cap


def test_min_freq_boundary():
    v = Vocab.from_captions([["four"] * 4, ["five"] * 5, ["six"] * 6], min_freq=5)
    assert "four" not in v      # count < min_freq is excluded
    assert "five" in v          # count == min_freq is kept (>=)
    assert "six" in v           # count > min_freq is kept


def test_keep_filter():
    v = Vocab.from_captions([[w] * 3 for w in ["dog", "cat1", "don't"]],
                            min_freq=0, keep=lambda w: w.isalnum() and w.islower())
    assert "dog" in v
    assert "cat1" in v
    assert "don't" not in v


def test_min_freq_filtering():
    caps = [["rare"] for _ in range(2)] + [["common"] for _ in range(4)]
    v = Vocab.from_captions(caps, min_freq=3)
    assert "common" in v
    assert "rare" not in v
    assert v.encode(["rare"]) == [v.stoi["<unk>"]]
    assert v.decode([v.stoi["<unk>"]]) == ["<unk>"]


def test_words_sorted_by_freq_desc():
    caps = [["z", "z", "a"], ["z", "b"], ["a"]]  # z:3, a:2, b:1
    v = Vocab.from_captions(caps, min_freq=0)
    assert v.stoi["z"] < v.stoi["a"] < v.stoi["b"]


def test_to_from_dict_round_trip():
    v = Vocab.from_captions(CAPS, min_freq=2)
    v2 = Vocab.from_dict(v.to_dict())
    assert v2.itos == v.itos
    assert v2.encode(["a", "dog"]) == v.encode(["a", "dog"])


def test_from_captions_cache_round_trip(tmp_path):
    v1 = Vocab.from_captions(CAPS, min_freq=1, cache_key="k1", cache_dir=tmp_path)
    files = list(tmp_path.glob("vocab_*.json"))
    assert len(files) == 1                      # miss: computed + written
    # hit (same key, wildly different captions -> cached vocab returned, not recounted)
    v2 = Vocab.from_captions([["zzz"]], min_freq=1, cache_key="k1", cache_dir=tmp_path)
    assert v2.itos == v1.itos
    # different key -> recomputed (own cache file)
    v3 = Vocab.from_captions(CAPS, min_freq=2, cache_key="k2", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("vocab_*.json"))) == 2
    assert v3.itos != v1.itos


def test_corrupt_cache_recomputes(tmp_path):
    import json
    key = "x"
    from captioning.vocab import Vocab as V
    import hashlib
    bogus = tmp_path / f"vocab_{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"
    bogus.write_text("not json")
    v = V.from_captions(CAPS, min_freq=1, cache_key=key, cache_dir=tmp_path)
    assert len(v) > 4 and bogus.exists()        # recomputed, cache overwritten