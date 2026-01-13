import optax
import jax
import jax.numpy as jnp
from tqdm import tqdm


def create_train_state():
    pass


def train_loop(model, train_state):

    @jax.jit
    def train_step(train_state):
        logits = model.apply(train_state.params)
        





