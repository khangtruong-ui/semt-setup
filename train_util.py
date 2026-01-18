import optax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from tqdm import tqdm
from flax.training.train_state import TrainState
from flax import serialization

from concurrent.futures import ThreadPoolExecutor

from config import *

mesh = Mesh(np.array(jax.devices()), ('data',))
sharding = NamedSharding(mesh, P('data'))
non_sharding = no_sharding = NamedSharding(mesh, P())

def save_checkpoint(params, i):
    target_bytes = serialization.to_bytes(params)
    with open(f'weights-{i:04d}.msgpack', 'wb') as f:
        f.write(target_bytes)

def create_train_state(model):
    imgs = jnp.zeros((BATCH_SIZE * jax.local_device_count(), 256, 256, 3))
    imgs = jax.device_put(imgs, sharding)
    captions = jnp.zeros((BATCH_SIZE * jax.local_device_count(), NUM_CAPTIONS, INPUT_SEQ_LENGTH), dtype=jnp.int32)
    captions = jax.device_put(captions, sharding)
    params = model.init(jax.random.key(0), (imgs, captions))
    tx = optax.adam(1e-4)
    train_state = TrainState.create(apply_fn=model.__call__, params=params, tx=tx)
    return train_state


def train_loop(model, train_state, ds, ds_length, epoches):

    @jax.jit
    def train_step(train_state, image, feed, label):

        @jax.value_and_grad
        def compute_loss(param):
            logits = model.apply(param, (image, feed))
            onehot_label = jax.nn.one_hot(label, VOCAB_SIZE)
            loss = optax.losses.softmax_cross_entropy(logits, onehot_label)
            loss = loss * (label != 0).astype(jnp.float32)
            return loss.sum() / loss.shape[0]

        loss, grad = compute_loss(train_state.params)
        new_state = train_state.apply_gradients(grads=grad)
        return loss, new_state

    def train_epoch(epoch, train_state, ds):
        with tqdm(total=ds_length, desc=f"Epoch {epoch}") as pbar:
            running_loss = 0.
            for i, item in zip(range(ds_length), ds):
                item = jax.tree.map(np.array, item)
                image = jax.device_put(item['image'], sharding)
                pad_left = jax.device_put(item['pad_left_ids'], sharding)
                pad_right = jax.device_put(item['pad_right_ids'], sharding)
                loss, train_state = train_step(train_state, image, pad_left, pad_right)
                running_loss = (running_loss * i + loss) / (i + 1)
                pbar.update(1)
                pbar.set_postfix({'loss': f"{running_loss:.4f}"})

        return train_state

    iter_ds = iter(ds)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for epoch in range(epoches):
            train_state = train_epoch(epoch, train_state, iter_ds)
            if (epoch + 1) % 5 == 0:
                executor.submit(save_checkpoint, train_state.params, epoch)

    return train_state
