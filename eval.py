from pycocoevalcap.eval import Bleu, Cider, Meteor, PTBTokenizer, Rouge, Spice
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import Constants as const
from models.base_captioner import BaseCaptioner
from utils.helper_functions import caption_array_to_string


DEBUG=False


def evaluate_caption_model(model: BaseCaptioner, dataset: Dataset) -> None:
    references = {}
    hypotheses = {}
    model.eval()
    
    print("Generating Captions for Test Set")
    iterator = tqdm(range(len(dataset)))
    for i in iterator:
        data = dataset.__getitem__(i)
        imgs = data[0].to(const.DEVICE)
        imgs = imgs.unsqueeze(0)
        reference_captions = data[1]
        image_id = dataset.ids[i]
        prediction = None

        if const.IS_GRAPH_MODEL:
            graphs = data[2]
            graphs[0].cuda()
            graphs[1].cuda()
            prediction = model.caption_image(graphs, dataset.vocab)
        else:
            prediction = model.caption_image(imgs, dataset.vocab)

        candidate_caption = caption_array_to_string(prediction)
        
        if DEBUG: 
            print(f"{image_id}: {candidate_caption}")

        # Collect the records we'll need for scoring
        hypotheses[image_id] = [{u'caption': candidate_caption}]
        references[image_id] = [{u'caption': caption} for caption in reference_captions]

    global_results, local_results = generate_scores(references, hypotheses)
    for k, v in global_results.items():
        print(f"{k}: {v:.3f}")


def generate_scores(references, candidates):
    tokenizer = PTBTokenizer()

    references  = tokenizer.tokenize(references)
    res = tokenizer.tokenize(candidates)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    output = {}
    img_output = {}

    for scorer, method in scorers:
        score, scores = scorer.compute_score(references, res)
        if type(method) != list:
            method = [method]
            score = [score]
            scores = [scores]

        for sc, scs, m in zip(score, scores, method):
            output[m] = sc
            for img_id, score in zip(references.keys(), scs):
                if type(score) is dict:
                    score = score['All']['f']

                if img_id not in img_output:
                    img_output[img_id] = {}
                img_output[img_id][m] = score

    return output, img_output