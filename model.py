import einops
import jax
import jax.numpy as jnp
import flax.linen as nn

from backbones.efficientnet import EfficientNetB1
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
        return out

# ==================================== SUPPORT CLASS =================================
class SelfAttention(nn.Module):
    @nn.compact
    def __call__(self, inp):
        length = inp.shape[-2]
        mask = jnp.arange(length)[..., None] >= jnp.arange(length)[None, ...]
        mask = mask[None, None, ...]
        minp = inp.reshape((-1, inp.shape[-2], inp.shape[-1]))
        mask = jnp.broadcast_to(mask, (minp.shape[0], ATTENTION_HEAD, length, length))
        out = MultiheadAttention()(minp, minp, minp, mask=mask).reshape(inp.shape)
        return out
        
# ==================================== VISION MODELS =================================
efficientnet = EfficientNetB1()

class EfficientNetVision(nn.Module):
    def preprocessing(self, x):
        x = jax.image.resize(x, (x.shape[0], 256, 256, 3), 'bicubic')
        x /= 255.
        mean = jnp.array([0.485, 0.456, 0.406])
        dev = jnp.array([0.229, 0.224, 0.225])
        return (x - mean) / dev
        
    def __call__(self, inp):
        x = self.preprocessing(inp)
        return efficientnet.forward(x)

class ShortVision(nn.Module):
    def preprocessing(self, x):
        x = jax.image.resize(x, (x.shape[0], 256, 256, 3), 'bicubic')
        x /= 255.
        mean = jnp.array([0.485, 0.456, 0.406])
        dev = jnp.array([0.229, 0.224, 0.225])
        return (x - mean) / dev

    @nn.compact
    def __call__(self, x):
        x = self.preprocessing(x)
        out = x
        out = nn.Conv(512, (2, 2), strides=(8, 8))(out)
        out = nn.Conv(1024, (2, 2), strides=(4, 4))(out)
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

class MultiLayerMeshed(nn.Module):
    def setup(self):
        self.enc = [MeshedEncoder() for _ in range(MESHED_DEPTH)]
        self.dec = [MeshedDecoder() for _ in range(MESHED_DEPTH)]

    def __call__(self, inp):
        src, tgt = inp
        srclst = [tgt]
        for block in self.enc:
            srclst.append(block(srclst[-1]))
        out = src
        for dec in self.dec:
            out = dec((out, srclst))
        return out

class NoMeshDecoder(nn.Module):
    def setup(self):
        self.sa = SelfAttention()
        self.ca = MultiheadAttention()
        self.norm = nn.LayerNorm()
        self.f = nn.Sequential([
            nn.Dense(TEXT_EMBEDDING_DIM),
            nn.activation.relu,
            nn.Dense(TEXT_EMBEDDING_DIM)
        ])

    def __call__(self, inp):
        src, tgt = inp
        sa = self.norm(self.sa(src))
        c = self.norm(self.ca(sa, tgt, tgt) + sa)
        f = self.norm(self.f(c) + c)
        return self.norm(f)

class MultiLayerNoMesh(nn.Module):
    def setup(self):
        self.enc = [MeshedEncoder() for _ in range(MESHED_DEPTH)]
        self.dec = [NoMeshDecoder() for _ in range(MESHED_DEPTH)]

    def __call__(self, inp):
        txt, img = inp
        for block in self.enc:
            img = block(img)
        x = txt
        for dec in self.dec:
            x = dec((x, img))
        return x

# ================================= MAIN MODULE =================================

class MeshedFastCaption(nn.Module):
    def setup(self):
        self.vision = {
            3: EfficientNetVision,
            5: ShortVision
        }[BACKBONE_CHOICE]()
        self.decoder = {
            0: MultiLayerMeshed,
            2: MultiLayerNoMesh,
        }[DECODER_ATTENTION_CHOICE]()
        self.adapt = nn.Dense(TEXT_EMBEDDING_DIM)
        self.dense = nn.Dense(VOCAB_SIZE)
        self.embedding = SeqEmbedding()

    def __call__(self, inputs):
        img, txt = inputs
        img = self.vision(img)
        img = self.adapt(img)
        img = img.reshape((img.shape[0], -1, img.shape[-1]))
        seq = self.embedding(txt)
        img = jnp.repeat(img[:, None, ...], seq.shape[1], axis=1)
        out = self.decoder((seq, img))
        return self.dense(out)

    def _batch_generate_from_index(self, imgs, txt, index):

        def cond(inp):
            index, txt = inp
            cond1 = index < txt.shape[2]
            unfinished_lines = (txt != 2).all(axis=-1)
            cond2 = unfinished_lines.any()
            return cond1 & cond2
        
        def loop(inp):
            index, txt = inp
            finished_lines = jnp.any(txt == 2, axis=-1)
            seq = self.embedding(txt)
            out = self.decoder((seq, imgs[:, None]))
            prob = self.dense(out)
            new_text = jnp.argmax(prob, axis=-1)
            valid = jnp.astype(jnp.arange(txt.shape[2]) <= index, jnp.int32)
            new_text = new_text * valid
            proposed_txt = jnp.concatenate([jnp.ones((new_text.shape[0], 1, 1), dtype=jnp.int32), new_text[..., :-1]], axis=-1)
            txt = jnp.where(finished_lines[..., None], txt, proposed_txt)
            index = index + 1
            return index, txt

        return jax.lax.while_loop(cond, loop, (0, txt))[1]

    def batch_generate_caption(self, imgs):
        imgs = self.vision(imgs)
        imgs = self.adapt(imgs)
        imgs = imgs.reshape((imgs.shape[0], -1, imgs.shape[-1]))
        txt = jnp.zeros((imgs.shape[0], 1, INPUT_SEQ_LENGTH), dtype=jnp.int32)
        out = self._batch_generate_from_index(imgs, txt, 0)
        return out

