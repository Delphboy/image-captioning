from .bleu import Bleu
from .cider import Cider
from .meteor import Meteor
from .rouge import Rouge
from .spice import Spice
from .tokenizer import PTBTokenizer


def compute_scores(gts, gen, is_test=False):
    if is_test:
        metrics = (Bleu(), Meteor(), Rouge(), Cider(), Spice())
    else:
        metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores
