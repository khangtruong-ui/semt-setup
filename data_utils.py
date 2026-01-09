import json
import numpy as np

def construct_tokenizer(ds):
    wordset = set()
    for item in ds:
        caption = item['caption']
        wordset.update(caption.split())

    wordlist = ['', '[BEGIN]', '[END]'] + sorted(list(wordset))
    tokenizer = {word: i for i, word in enumerate(wordlist)}
    with open('tokenizer.json', 'w') as f:
        json.dump(tokenizer, f)


def load_tokenizer():
    with open('tokenizer.json') as f:
        return json.load(f)


def sentences_mapper(tokenizer, max_length):
    def mapping(item):
        caption = item['caption']
        ids = [tokenizer.get(word) for word in caption.split()]
        ids.extend([0] * (max_length - len(ids)))
        item['input_ids'] = ids
        return item

