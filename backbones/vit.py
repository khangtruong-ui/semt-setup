import flax.linen as nn
from transformers import FlaxCLIPModel
import jax
import jax.numpy as jnp


class Vit(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        vision_model, vision_params = crafted_model()
        output = vision_model.apply(vision_params, x)
        return output['last_hidden_state']

crafted_model = None

def get_backbone_and_weight():
    global crafted_model
    if crafted_model is not None:
        return crafted_model
    clip = FlaxCLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    module = clip.module
    variables = {'params': clip.params}
    clip_bind = module.bind(variables)
    vision_model, vision_params = clip_bind.vision_model.unbind()
    return vision_model, vision_params
