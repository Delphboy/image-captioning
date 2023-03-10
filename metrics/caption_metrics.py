from nltk.translate.bleu_score import corpus_bleu


def bleu_score(ground_truths, predicted):
    bleu_dic = {}
    bleu_dic['1-grams'] = corpus_bleu(ground_truths, predicted, weights=(1.0, 0, 0, 0))
    bleu_dic['1-2-grams'] = corpus_bleu(ground_truths, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_dic['1-3-grams'] = corpus_bleu(ground_truths, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu_dic['1-4-grams'] = corpus_bleu(ground_truths, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    
    return bleu_dic