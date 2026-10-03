"""Model registry: name -> class; build via MODELS[cfg['model']['name']]."""
from .transformer import Transformer

MODELS = {"transformer": Transformer}


def build(cfg: dict, vocab) -> Transformer:
    """Instantiate the model named in cfg['model'], sized by the vocab + data cfg."""
    m = dict(cfg["model"])
    name = m.pop("name")
    return MODELS[name](len(vocab), max_len=cfg["data"]["max_cap_len"], **m)