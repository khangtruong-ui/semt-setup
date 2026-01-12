import einops
import jax
import jax.numpy as jnp
import flax.linen as nn

from config import *

# =========================== UTILITIES ====================================
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

# ================================ ATTENTION THINGS ====================================
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

class MultiheadMemorizedAttention(nn.Module):
    def setup(self):
        self.heads = [MemorizedAttention(TEXT_EMBEDDING_DIM // ATTENTION_HEAD) for _ in range(ATTENTION_HEAD)]
        self.norm = nn.LayerNorm()
        self.dense = nn.Dense(TEXT_EMBEDDING_DIM)

    def __call__(self, q, k, v):
        tensors = [head(q, k, v) for head in self.heads]
        return self.norm(self.dense(jnp.concatenate(tensors, axis=-1)) + q)

class MultiheadAttention(nn.Module):
    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=ATTENTION_HEAD, 
            qkv_features=TEXT_EMBEDDING_DIM,
            out_features=TEXT_EMBEDDING_DIM,
        )

    def __call__(self, q, k, v):
        return self.attn(q, k, v)

class MultiheadStaticAttention(nn.Module):
    def setup(self):
        self.Q = nn.Dense(TEXT_EMBEDDING_DIM)
        self.V = nn.Dense(TEXT_EMBEDDING_DIM)
        self.norm = nn.LayerNorm(TEXT_EMBEDDING_DIM)
        self.memory = self.param('memory', 
                                 nn.initializers.glorot_uniform(), 
                                 (EXPANSION_LENGTH, ATTENTION_HEAD, TEXT_EMBEDDING_DIM // ATTENTION_HEAD))

    def __call__(self, q, k, v):
        batch = q.shape[0]
        length = q.shape[2]
        dim = q.shape[3]
        Q_ = self.Q(q)
        V_ = self.V(k)
        Q = Q_.reshape(batch, length, ATTENTION_HEAD, dim // ATTENTION_HEAD)
        V = V_.reshape(batch, length, ATTENTION_HEAD, dim // ATTENTION_HEAD)
        eij = jnp.einsum('ehd, blhd -> belh', self.memory, Q)
        eij = jnp.where(eij > 0, eij, 0.)
        out1 = jnp.einsum('belh, blhd -> behd', eij, V)
        out2 = jnp.einsum('behd, belh -> blhd', out1, eij)
        out = self.norm(out2.reshape(batch, 1, length, dim))

# ==================================== ENCODERS DECODERS =============================
class MeshedEncoder(nn.Module):
    pass






