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
import os
import re

from data_utils import get_train_set, get_test_set, load_tokenizer
from model import MeshedFastCaption
from train_util import create_train_state

mesh = Mesh(np.array(jax.devices()), ('data',))
sharding = NamedSharding(mesh, P('data'))
non_sharding = no_sharding = NamedSharding(mesh, P())

compiled_inference_function = None

def load_checkpoint(model_index=-1):
    save_dir = os.environ['SAVE_DIR']
    fname = sorted(glob.glob(save_dir + '/*.msgpack'))[model_index]
    int_string = re.findall(r'\d+', fname)[0]
    print(f'USED WEIGHT: {fname}')
    model = MeshedFastCaption()
    state = create_train_state(model)
    with open(fname, 'rb') as f:
        return serialization.from_bytes(state.params, f.read()), int_string

def reverse_tensor(tens):
    tokenizer = load_tokenizer()
    reversed_tokenizer = {tokenizer[k]: k for k in tokenizer}
    
    def mapper(cap_int):
        assert cap_int.ndim == 1, f"cap_int shape: {cap_int.shape}"
        return ' '.join(reversed_tokenizer.get(int(token), '') for token in cap_int).strip().replace('[END] [END]', '[END]')

    collected = np.array([[mapper(cap) for cap in cap_per_batch] for cap_per_batch in tens]).reshape(tens.shape[:-1])
    return collected

def inference(model, weights, weights_name):
    global compiled_inference_function
    test_set, ds_length = get_test_set()

    @jax.jit
    def loop_body(image, weights):
        print('Tracing loop_body', image.shape, jax.tree.map(jnp.shape, weights))
        out = model.apply(weights, image, method=model.batch_generate_caption)
        return out

    compiled_inference_function = compiled_inference_function if compiled_inference_function is not None loop_body
    
    ret_dict = []
    res_dict = []
    for _, batch in zip(tqdm(range(ds_length)), test_set):
        batch = jax.tree.map(np.array, batch)
        image = jax.device_put(batch['image'], sharding)
        out = compiled_inference_function(image, weights)
        out_sentence = reverse_tensor(out)[:, 0]
        out_caption = reverse_tensor(batch['pad_right_ids'])
        ret_dict.extend(out_sentence.tolist())
        res_dict.extend(out_caption.tolist())

    os.makedirs(os.environ['INFERENCE_DIR'], exist_ok=True)
    with open(f"{os.environ['INFERENCE_DIR']}/predict-{weights_name}.json", 'w') as f:
        json.dump(ret_dict, f)

    with open(f"{os.environ['INFERENCE_DIR']}/label-{weights_name}.json", 'w') as f:
        json.dump(res_dict, f)
        
def main():
    model = MeshedFastCaption()
    for item in range(len(os.listdir(os.environ['SAVE_DIR']))):
        weights, fname = load_checkpoint(item)
        inference(model, weights, fname)

if __name__ == '__main__':
    main()




