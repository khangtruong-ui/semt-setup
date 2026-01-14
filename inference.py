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

mesh = Mesh(np.array(jax.devices()), ('data',))
sharding = NamedSharding(mesh, P('data'))
non_sharding = no_sharding = NamedSharding(mesh, P())


def load_checkpoint():
    fname = sorted(glob.glob('*.msgpack'))[-1]
    with open('weights.msgpack', 'rb') as f:
        return serialization.from_bytes(f.read())

def reverse_tensor(tens):
    tokenizer = load_tokenizer()
    
    def mapper(cap_int):
        return ' '.join(tokenizer[token] for token in cap_int).strip().replace('[END] [END]', '[END]')

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
    for _, batch in zip(range(100), test_set):
        image = jax.device_put(batch['image'], sharding)
        out = loop_body(image)
        out_sentence = reverse_tensor(out)
        out_caption = reverse_tensor(batch['pad_right_ids'])
        print(out_caption)
        print(out_sentence)
        ret_dict.append(out_sentence)
        res_dict.append(out_caption)

    with open('predict.json', 'w') as f:
        json.dump(ret_dict, f)

    with open('label.json', 'w') as f:
        json.dump(res_dict, f)
        
def main():
    model = MeshedFastCaption()
    weights = load_checkpoint()
    inference(model, weights)

if __name__ == '__main__':
    main()




