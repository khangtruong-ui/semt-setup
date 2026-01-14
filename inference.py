import optax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from tqdm import tqdm
from flax import serialization
import glob


def load_checkpoint():
    fname = sorted(glob.glob('*.msgpack'))[-1]
    with open('weights.msgpack', 'rb') as f:
        return serialization.from_bytes(f.read())

def inference(model, weights):
    pass
