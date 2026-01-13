import optax
import jax
import jax.numpy as jnp
from tqdm import tqdm
from flax.training import TrainState

from config import *


def create_train_state(model):
    imgs = jnp.zeros((BATCH_SIZE, 256, 256, 3))
    captions = jnp.zeros((BATCH_SIZE, NUM_CAPTIONS, INPUT_SEQ_LENGTH), dtype=jnp.int32)
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
                caption = item['input_ids']
                loss, train_state = train_step(train_state, image, caption)
                pbar.update(1)
                pbar.set_postfix({'loss': (running_loss * i + loss) / (i + 1)})

        return train_state

    for epoch in range(epoches):
        train_state = train_epoch(epoch, train_state, ds)

    return train_state





