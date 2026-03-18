import json
import numpy as np
import os
from datasets import load_dataset, load_from_disk
from PIL import Image
import grain.python as grain
import jax
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from config import *


def construct_tokenizer(ds):
    wordset = set()
    for item in ds:
        key = 'caption' if 'caption' in item else 'raw'
        caption = item[key]
        wordset.update(caption.lower().split())

    wordlist = ['', '[BEGIN]', '[END]'] + sorted(list(wordset))
    tokenizer = {word: i for i, word in enumerate(wordlist)}
    with open('tokenizer.json', 'w') as f:
        json.dump(tokenizer, f)


def load_tokenizer():
    with open('tokenizer.json') as f:
        return json.load(f)

def sentences_mapper(tokenizer, max_length=MAX_LENGTH):
    def mapping(item):
        item['pad_left_ids'], item['pad_right_ids'] = [], []
        
        for key in ['caption', 'raw', 'raw_1', 'raw_2', 'raw_3', 'raw_4']:
            if key not in item:
                continue
                
            caption = item[key]
            
            def select(word):
                if word in tokenizer:
                    return tokenizer[word]
                else:
                    # print('Unknown word:', word.encode())
                    return 0
            
            ids = [select(word) for word in caption.lower().split()]
            pad_left_ids = [1] + ids
            pad_right_ids = ids + [2]
            pad_left_ids.extend([0] * (max_length - len(pad_left_ids)))
            pad_right_ids.extend([0] * (max_length - len(pad_right_ids)))
            item['pad_left_ids'].append(pad_left_ids)
            item['pad_right_ids'].append(pad_right_ids)

        return {k: np.array(item[k]) for k in item}

    return mapping

class TorchDataset(Dataset):
    def __init__(self, ds, mapper):
        self.ds = ds
        self.mapper = mapper

    def __len__(self):
        return len(self.ds) * 10000

    def __getitem__(self, i):
        item = self.ds[i % len(self.ds)]
        mapped = self.mapper(item)
        ret = dict(
            image=mapped['image'],
            pad_left_ids=mapped['pad_left_ids'],
            pad_right_ids=mapped['pad_right_ids']
        )
        output = jax.tree.map(lambda x: np.array(x), ret)
        assert jax.tree.all(jax.tree.map(lambda x: x.dtype != np.dtype(object), output)), f"Type: {jax.tree.map(lambda x: x.dtype, output)}\nOutput: {output}"
        
        return output

def get_set(ds, batch_size=BATCH_SIZE):
    sampler = grain.IndexSampler(
        num_records=len(ds),
        shard_options=grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            drop_remainder=True,
        ),
        shuffle=True,
        seed=42,
    )
    
    if not os.path.exists('tokenizer.json'):
        construct_tokenizer(ds)
    tokenizer = load_tokenizer()
    sentence_map = sentences_mapper(tokenizer)
    mapped_ds = TorchDataset(ds, sentence_map)
    
    loader = grain.DataLoader(
        data_source=mapped_ds,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size * jax.local_device_count(), drop_remainder=True)],
    )
    torch_sampler = DistributedSampler(
        dataset=mapped_ds,
        num_replicas=jax.process_count(),   # == num_programs
        rank=jax.process_index(),                 # == program_index
        shuffle=True,
    )

    torch_loader = DataLoader(mapped_ds, 
                              batch_size=batch_size * jax.local_device_count(),
                              sampler=torch_sampler,
                              drop_last=True,
                              num_workers=os.cpu_count() // 2
                             )
    
    loader_length = len(ds) // jax.local_device_count() // batch_size
    return torch_loader, loader_length

def get_train_set():
    return get_set(load_dataset(os.environ['TRAIN_DATASET'])['train'].with_format('np'), batch_size=BATCH_SIZE)

def get_test_set():
    ds = load_dataset(os.environ['TEST_DATASET'])
    ds = ds['test'] if 'test' in ds else ds['train']
    return get_set(ds.with_format('np'), batch_size=BATCH_SIZE * 16)




    
