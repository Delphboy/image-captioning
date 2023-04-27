from constants import Constants as const
from utils.helper_functions import caption_array_to_string

from metrics.caption_metrics import bleu_score
from torch.utils.data import Dataset
from tqdm import tqdm
from models.base_captioner import BaseCaptioner

DEBUG=True

def evaluate_graph_caption_model(model: BaseCaptioner,
                           dataset: Dataset):
    references = []
    hypotheses = []
    img_idx = 0
    model.eval()
    print("Generating Captions for Test Set")
    for data in iter(dataset):
        imgs = data[0].to(const.DEVICE)
        imgs = imgs.unsqueeze(0)
        reference_captions = data[1]
        graphs = data[2]
        graphs[0].cuda()
        graphs[1].cuda()

        prediction = model.caption_image(graphs, dataset.vocab)
        candidate_caption = caption_array_to_string(prediction)
        # if DEBUG: print(f"{dataset.dataset.imgs[dataset.indices[img_idx]]}: {candidate_caption}")
        if DEBUG: print(f"{dataset.ids[img_idx]}: {candidate_caption}")
        
        hypotheses.append(candidate_caption)
        references.append(reference_captions)
        img_idx += 1
        

    print("Calculating BLEU Score")
    print(bleu_score(references, hypotheses))



def evaluate_caption_model(model: BaseCaptioner,
                           dataset: Dataset):
    references = []
    hypotheses = []
    
    img_idx = 0
    model.eval()
    print("Generating Captions for Test Set")
    for data in iter(dataset):
        imgs = data[0].to(const.DEVICE)
        imgs = imgs.unsqueeze(0)
        reference_captions = data[1]

        prediction = model.caption_image(imgs, dataset.vocab)
        candidate_caption = caption_array_to_string(prediction)
        # if DEBUG: print(f"{dataset.dataset.imgs[dataset.indices[img_idx]]}: {candidate_caption}")
        if DEBUG: print(f"{dataset.ids[img_idx]}: {candidate_caption}")
        
        hypotheses.append(candidate_caption)
        references.append(reference_captions)
        img_idx += 1
        

    print("Calculating BLEU Score")
    print(bleu_score(references, hypotheses))
