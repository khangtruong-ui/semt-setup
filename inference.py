print('===== INFERENCE SCRIPT =====')

import optax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from tqdm import tqdm
from flax import serialization
import glob
import json

from data_utils import get_train_set, load_tokenizer
from model import MeshedFastCaption
from train_util import create_train_state

mesh = Mesh(np.array(jax.devices()), ('data',))
sharding = NamedSharding(mesh, P('data'))
non_sharding = no_sharding = NamedSharding(mesh, P())


def load_checkpoint():
    fname = sorted(glob.glob('*.msgpack'))[-1]
    print('====== CRAFTING MODELS ======')
    print(f'USED WEIGHT: {fname}')
    model = MeshedFastCaption()
    state = create_train_state(model)
    with open(fname, 'rb') as f:
        return serialization.from_bytes(state.params, f.read())

def reverse_tensor(tens):
    tokenizer = load_tokenizer()
    reversed_tokenizer = {tokenizer[k]: k for k in tokenizer}
    
    def mapper(cap_int):
        assert cap_int.ndim == 1, f"cap_int shape: {cap_int.shape}"
        return ' '.join(reversed_tokenizer[int(token)] for token in cap_int).strip().replace('[END] [END]', '[END]')

    collected = np.array([[mapper(cap) for cap in cap_per_batch] for cap_per_batch in tens]).reshape(tens.shape[:-1])
    return collected

def inference(model, weights):
    test_set = get_train_set()

    @jax.jit
    def loop_body(image):
        out = model.apply(weights, image, method=model.batch_generate_caption)
        return out

    ret_dict = []
    res_dict = []
    for _, batch in zip(range(1), test_set):
        batch = jax.tree.map(np.array, batch)
        image = jax.device_put(batch['image'], sharding)
        out = loop_body(image)
        out_sentence = reverse_tensor(out)
        out_caption = reverse_tensor(batch['pad_right_ids'])
        print(out_caption)
        print(out_sentence)
        ret_dict.append(out_sentence.tolist())
        res_dict.append(out_caption.tolist())

    with open('predict.json', 'w') as f:
        json.dump(ret_dict, f)

    with open('label.json', 'w') as f:
        json.dump(res_dict, f)
        
def main():
    model = MeshedFastCaption()
    weights = load_checkpoint()
    print('===== INFERENCING =====')
    inference(model, weights)

if __name__ == '__main__':
    main()




