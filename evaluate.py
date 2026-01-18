import json
from aac_metrics import evaluate
import pandas as pd

def evaluate_from_files():
    with open('predict.json') as f1, open('label.json') as f2:
        prediction = json.load(f1)
        references = json.load(f2)
    result = evaluate(prediction, references, metrics=['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge_l'])[0]
    dictionary = {k: [v.cpu().item()] for k, v in result.items()}
    csv_result = pd.DataFrame.from_dict(dictionary)
    print(csv_result)

if __name__ == "__main__":
    evaluate_from_files()
