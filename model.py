import einops
import jax
import jax.numpy as jnp
import flax.linen as nn
import eqxvision as eqv

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

    def __call__(self, q, k, v, mask=None):
        return self.attn(q, k, v, mask=mask)

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

# ==================================== SUPPORT CLASS =================================
class SelfAttention(nn.Module):
    @nn.compact
    def __call__(self, inp):
        length = inp.shape[-2]
        mask = jnp.arange(length)[..., None] >= jnp.arange(length)[None, ...]
        mask = jnp.broadcast_to(mask, (inp.shape[0], ATTENTION_HEAD, length, length))
        return MultiheadAttention()(inp, inp, inp, mask=mask)

# ==================================== VISION MODELS =================================
efficientnetb2 = eqv.models.classification.efficientnet_b2('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth')

class EfficientNetVision(nn.Module):
    def preprocessing(self, x):
        mean = jnp.array([0.485, 0.456, 0.406])
        dev = jnp.array([0.229, 0.224, 0.225])
        return (x - mean) / dev
        
    def __call__(self, x):
        x = self.preprocessing(x)
        out = efficientnetb2.features(x, key=jax.random.key(0))
        out = efficientnetb2.avgpool(x)
        return out

# ==================================== ENCODERS DECODERS =============================
class MeshedEncoder(nn.Module):
    def setup(self):
        self.m_attn = {
            0: MultiheadMemorizedAttention,
            2: MultiheadAttention,
            3: MultiheadStaticAttention
        }[ATTENTION_CHOICE]()
        self.f = nn.Sequential([
            nn.Dense(TEXT_EMBEDDING_DIM),
            nn.activation.relu,
            nn.Dense(TEXT_EMBEDDING_DIM),
        ])
        self.norm = nn.LayerNorm()

    def __call__(self, inp):
        z = self.norm(self.m_attn(inp, inp, inp) + inp)
        x = self.norm(self.f(z) + z)
        return x

class MeshedDecoder(nn.Module):
    def setup(self):
        self.sa = SelfAttention()
        self.ca = MultiheadAttention()
        self.dense = nn.Dense(TEXT_EMBEDDING_DIM)
        self.norm = nn.LayerNorm()
        self.f = nn.Sequential([
            nn.Dense(TEXT_EMBEDDING_DIM),
            nn.activation.relu,
            nn.Dense(TEXT_EMBEDDING_DIM)
        ])

    def __call__(self, inp):
        src, tgts = inp
        sa = self.norm(self.sa(src))
        gated = jnp.zeros_like(sa)
        for tgt in tgts:
            c = self.norm(self.ca(sa, tgt, tgt) + sa)
            alpha = self.dense(jnp.concatenate([sa, c], axis=-1))
            alpha = jax.nn.sigmoid(alpha)
            feed = alpha * c
            gated += feed
        f = self.norm(self.f(gated) + gated)
        return self.norm(f)





