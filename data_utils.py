import json
import numpy as np
import os
from datasets import load_dataset

from config import *


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

def sentences_mapper(tokenizer, max_length=MAX_LENGTH):
    def mapping(item):
        caption = item['caption']
        ids = [tokenizer.get(word) for word in caption.split()]
        if pad_left:
            pad_left_ids = [1] + ids
        if pad_right:
            pad_right_ids = ids + [2]
        pad_left_ids.extend([0] * (max_length - len(pad_left_ids)))
        pad_right_ids.extend([0] * (max_length - len(pad_right_ids)))
        item['pad_left_ids'] = pad_left_ids
        item['pad_right_ids'] = pad_right_ids
        return item

    return mapping

def get_set(ds):
    if not os.path.exists('tokenizer.json'):
        construct_tokenizer(ds)
    tokenizer = load_tokenizer()
    sentence_map = sentence_mapper(tokenizer)
    mapped_ds = ds.map(sentence_map, load_from_cache=False).batch(BATCH_SIZE, load_from_cache=False, drop_last_batch=True)
    return mapped_ds

def get_train_set():
    return get_set(load_dataset(os.environ['DATASET']).with_format('np')['train'])

def get_test_set():
    return get_set(load_dataset(os.environ['DATASET']).with_format('np')['test'])



    
