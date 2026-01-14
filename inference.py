import optax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from tqdm import tqdm
from flax import serialization
import glob

from data_utils import get_train_set


def load_checkpoint():
    fname = sorted(glob.glob('*.msgpack'))[-1]
    with open('weights.msgpack', 'rb') as f:
        return serialization.from_bytes(f.read())

def inference(model, weights):
    test_set = get_train_set()
    for _, batch in zip(range(100), test_set):
        image = batch['image']
        out = model.apply(weights, model.batch_generate_caption)




