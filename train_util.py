import optax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tqdm import tqdm
from flax.training.train_state import TrainState
from flax import serialization

from config import *

mesh = Mesh(np.array(jax.devices()), ('data',))
sharding = NamedSharding(mesh, P('data'))
non_sharding = no_sharding = NamedSharding(mesh, P())

def save_checkpoint(params):
    target_bytes = serialization.to_bytes(params)
    with open('weights.msgpack', 'wb') as f:
        f.write(target_bytes)

def load_checkpoint():
    with open('weights.msgpack', 'rb') as f:
        return serialization.from_bytes(f.read())

def create_train_state(model):
    imgs = jnp.zeros((BATCH_SIZE, 256, 256, 3))
    imgs = jax.device_put(imgs, sharding)
    captions = jnp.zeros((BATCH_SIZE, NUM_CAPTIONS, INPUT_SEQ_LENGTH), dtype=jnp.int32)
    captions = jax.device_put(captions, sharding)
    params = model.init(jax.random.key(0), (imgs, captions))
    tx = optax.adam(1e-4)
    train_state = TrainState.create(apply_fn=model.__call__, params=params, tx=tx)
    return train_state


def train_loop(model, train_state, ds, epoches):

    @jax.jit
    def train_step(train_state, image, feed, label):

        @jax.value_and_grad
        def compute_loss(param):
            logits = state.apply(param, image, feed)
            loss = optax.losses.softmax_cross_entropy(logits, label)
            return loss

        loss, grad = compute_loss(train_state.params)
        new_state = train_state.apply_gradients(grad)
        return loss, new_state

    def train_epoch(epoch, train_state, ds):
        with tqdm(total=len(ds), desc=f"Epoch {f}") as pbar:
            running_loss = 0.
            for i, item in enumerate(ds):
                image = item['image']
                pad_left = item['pad_left_ids']
                pad_right = item['pad_right_ids']
                loss, train_state = train_step(train_state, image, pad_left, pad_right)
                pbar.update(1)
                pbar.set_postfix({'loss': (running_loss * i + loss) / (i + 1)})

        return train_state

    for epoch in range(epoches):
        train_state = train_epoch(epoch, train_state, ds)
        if epoch % 1 == 0:
            save_checkpoint(train_state.params)

    return train_state
