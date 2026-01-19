import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from functools import reduce
import json

nltk.download('wordnet')

def compute_metric_mapping(predicted, corpus):
    candidate = predicted.split()
    references = [x.split() for x in corpus]

    # BLEU-1 score (unigram)
    bleu_1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0))
    
    # BLEU-2 score (bigram)
    bleu_2 = sentence_bleu(references, candidate, weights=(0.5, 0.5, 0, 0))
    
    # BLEU-3 score (trigram)
    bleu_3 = sentence_bleu(references, candidate, weights=(0.33, 0.33, 0.34, 0))
    
    # BLEU-4 score (quadrigram)
    bleu_4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    meteor = meteor_score(references, candidate)
    
    return dict(
        bleu_1=bleu_1,
        bleu_2=bleu_2,
        bleu_3=bleu_3,
        bleu_4=bleu_4,
        meteor=meteor,
    )
    

def main():
    with open('predict.json') as p, open('label.json') as l:
        predicted = json.load(p)
        label = json.load(l)

    def sum_reducer(metrics, new_metrics):
        if metrics is None:
            return new_metrics
    
        return {k: metrics[k] + new_metrics[k] for k in metrics}

    metr = [compute_metric_mapping(u, v) for u, v in zip(predicted, label)]
    summed = reduce(sum_reducer, metr, None)
    meant = {k: v / len(metr) for k, v in summed.items()}
    print(meant)
    return meant
    

if __name__ == '__main__':
    main()
    










