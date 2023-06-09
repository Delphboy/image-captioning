from typing import Tuple
from pycocoevalcap.eval import Bleu, Cider, Meteor, PTBTokenizer, Rouge, Spice
from torch.utils.data import Dataset
from constants import Constants as const
from models.base_captioner import BaseCaptioner
from utils.helper_functions import caption_array_to_string
from torch_geometric.data import Batch


DEBUG=False


def evaluate_caption_model(model: BaseCaptioner, dataset: Dataset) -> Tuple[dict, dict]:
    references = {}
    hypotheses = {}
    model.eval()
    
    print("Generating Captions")
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        imgs = data[0].to(const.DEVICE)
        imgs = imgs.unsqueeze(0)
        reference_captions = data[1]
        image_id = dataset.ids[i]
        prediction = None

        if const.IS_GRAPH_MODEL:
            graphs = data[2]
            graphs = [Batch.from_data_list([graph.cuda()]) for graph in graphs]
            prediction = model.caption_image(graphs, dataset.vocab)
        else:
            prediction = model.caption_image(imgs, dataset.vocab)

        candidate_caption = caption_array_to_string(prediction)
        
        if DEBUG: 
            print(f"{image_id}: {candidate_caption}")

        # Collect the records we'll need for scoring
        hypotheses[str(image_id)] = [{u'caption': candidate_caption}]
        references[str(image_id)] = [{u'caption': caption} for caption in reference_captions]

    global_results, local_results = generate_scores(references, hypotheses)
    for k, v in global_results.items():
        v*=100
        print(f"{k}: {v:.3f}")
    
    return global_results, local_results


def generate_scores(references, candidates):
    tokenizer = PTBTokenizer()

    references  = tokenizer.tokenize(references)
    res = tokenizer.tokenize(candidates)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE") # WARNING: This can be blocking with multiple instances of the captioning code running
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

