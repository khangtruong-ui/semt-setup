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
        pad_left_ids = [1] + ids
        pad_right_ids = ids + [2]
        pad_left_ids.extend([0] * (max_length - len(pad_left_ids)))
        pad_right_ids.extend([0] * (max_length - len(pad_right_ids)))
        item['pad_left_ids'] = np.repeat(np.array(pad_left_ids)[None, ...], NUM_CAPTIONS, axis=0)
        item['pad_right_ids'] = np.repeat(np.array(pad_right_ids)[None, ...], NUM_CAPTIONS, axis=0)

        return item

    return mapping

class TorchDataset(Dataset):
    def __init__(self, ds, mapper):
        self.ds = ds
        self.mapper = mapper

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        mapped = self.mapper(item)
        ret = dict(
            image=mapped['image'],
            pad_left_ids=mapped['pad_left_ids'],
            pad_right_ids=mapped['pad_right_ids']
        )
        return jax.tree.map(lambda x: np.array(x).astype(np.int32), ret)

def get_set(ds):
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
    # ds = grain.MapDataset.source(ds)
    # mapped_ds = ds.map(sentence_map).map(lambda x: jax.tree.map(np.array, x))
    mapped_ds = TorchDataset(ds, sentence_map)
    
    loader = grain.DataLoader(
        data_source=mapped_ds,
        sampler=sampler,
        operations=[grain.Batch(batch_size=BATCH_SIZE * jax.local_device_count(), drop_remainder=True)],
    )
    torch_sampler = DistributedSampler(
        dataset=mapped_ds,
        num_replicas=jax.process_count(),   # == num_programs
        rank=jax.process_index(),                 # == program_index
        shuffle=True,
    )

    torch_loader = DataLoader(mapped_ds, 
                              batch_size=BATCH_SIZE * jax.local_device_count(),
                              sampler=torch_sampler,
                              drop_last=True,
                              num_workers=os.cpu_count() // 2
                             )
    return torch_loader

def get_train_set():
    return get_set(load_dataset(os.environ['DATASET'])['train'].with_format('np'))




    
