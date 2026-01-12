import einops
import jax
import jax.numpy as jnp
import flax.linen as nn

from config import *


class Mask(nn.Module):
    def __call__(self, inputs):
        length = inputs.shape[-2]
        mask = jnp.arange(length)[..., None] >= jnp.arange(length)[None, ...]
        return jnp.where(mask, 0., -1e9)


class SeqEmbedding(nn.Module):
    def setup(self):
        self.token_emb = nn.Embed(VOCAB_SIZE, TEXT_EMBEDDING_DIM)
        self.position = nn.Embed(INPUT_SEQ_LENGTH, TEXT_EMBEDDING_DIM)

    def __call__(self, txt):
        return self.token_emb(txt) + self.position(jnp.arange(txt.shape[-1]))[None, ...]


class MemorizedAttention(nn.Module):
    dim: int
    def setup(self):
        self.Q = nn.Dense(self.dim)
        self.K = nn.Dense(self.dim)
        self.V = nn.Dense(self.dim)

    def __call__(self, q, k, v):
        repeater = q.shape[:-2] + (1, 1)
        memory_k = jnp.tile(self.memory_k, repeater)
        memory_v = jnp.tile(self.memory_v, repeater)
        Q = self.Q(q)
        K = jnp.concatenate([self.K(k), memory_k], axis=-2)
        V = jnp.concatenate([self.V(v), memory_v], axis=-2)
        attn = jnp.einsum('...qd, ...kd -> ...qk', Q, K) / jnp.sqrt(TEXT_EMBEDDING_DIM)
        attn = jax.nn.softmax(attn, axis=-1)
        return jnp.einsum('...qk, ...kd -> ...qd', attn, V)







