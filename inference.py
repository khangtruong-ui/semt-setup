import optax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from tqdm import tqdm
from flax import serialization
import glob

from data_utils import get_train_set, load_tokenizer


def load_checkpoint():
    fname = sorted(glob.glob('*.msgpack'))[-1]
    with open('weights.msgpack', 'rb') as f:
        return serialization.from_bytes(f.read())

def reverse_tensor(tens):
    tokenizer = load_tokenizer()
    for pred in tens:
        cap_int = pred[0]
        cap_int = 

def inference(model, weights):
    test_set = get_train_set()

    @jax.jit
    def loop_body(batch):
        image = batch['image']
        out = model.apply(weights, image, method=model.batch_generate_caption)
        return out

    for _, batch in zip(range(100), test_set):
        out = loop_body(batch)
        




