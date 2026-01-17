import flax.linen as linen
import jax.numpy as jnp
import jax
import math
import re
import tensorflow as tf
import keras


def round_filters(filters, width_coeff, divisor=8):
  """Round number of filters based on width multiplier."""
  filters *= width_coeff
  new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)

class Normalization(linen.Module):
  @linen.compact
  def __call__(self, x):
    mean = self.variable('params', 'mean', lambda: jnp.zeros((1, 1, 1, 3)))
    variance = self.variable('params', 'variance', lambda: jnp.ones((1, 1, 1, 3)))
    count = self.variable('params', 'count', lambda: jnp.array(1.0))
    x = (x - mean.value) / jnp.sqrt(variance.value + 1e-5)
    return x

class DropPathState(linen.Module):
  @linen.compact
  def __call__(self, x):
    self.variable('params', 'seed_generator_state', lambda: jnp.array(0, dtype=jnp.uint32))
    return x

class MBConv(linen.Module):
  expand_ratio: int
  kernel_size: tuple[int, int]
  strides: tuple[int, int]
  in_channels: int
  out_channels: int
  se_ratio: float = 0.25
  drop_rate: float = 0.0

  @linen.compact
  def __call__(self, x, train: bool):
    out = x
    expanded = self.expand_ratio * self.in_channels
    if self.expand_ratio != 1:

        x = linen.Conv(expanded, kernel_size=(1, 1), padding='SAME', use_bias=False, name='expand_conv')(x)
        x = linen.BatchNorm(use_running_average=not train, name='expand_bn')(x)
        x = linen.swish(x)

    x = linen.Conv(expanded, kernel_size=self.kernel_size, strides=self.strides, padding='SAME', use_bias=False, feature_group_count=expanded, name='dwconv')(x)
    x = linen.BatchNorm(use_running_average=not train, name='bn')(x)
    x = linen.swish(x)

    # Squeeze and Excitation
    se = jnp.mean(x, axis=[1, 2], keepdims=True)
    se = linen.Conv(max(1, int(self.in_channels * self.se_ratio)), kernel_size=(1, 1), padding='SAME', use_bias=True, name='se_reduce')(se)
    se = linen.swish(se)
    se = linen.Conv(expanded, kernel_size=(1, 1), padding='SAME', use_bias=True, name='se_expand')(se)
    x = x * linen.sigmoid(se)

    x = linen.Conv(self.out_channels, kernel_size=(1, 1), padding='SAME', use_bias=False, name='project_conv')(x)
    x = linen.BatchNorm(use_running_average=not train, name='project_bn')(x)

    if self.strides == (1, 1) and self.in_channels == self.out_channels:
      if self.drop_rate > 0:
        x = linen.Dropout(rate=self.drop_rate, deterministic=not train)(x)
      x = x + out
    return x

class EfficientNetB1(linen.Module):
  @linen.compact
  def __call__(self, x, train: bool = False):
    width_coeff = 1.1
    depth_coeff = 1.2

    x = Normalization(name='normalization')(x)

    stem_channels = round_filters(32, width_coeff)
    x = linen.Conv(stem_channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME', use_bias=False, name='stem_conv')(x)
    x = linen.BatchNorm(use_running_average=not train, name='stem_bn')(x)
    x = linen.swish(x)

    blocks_args = [
      {'num_repeat': round(math.ceil(1 * depth_coeff)), 'kernel_size': (3, 3), 'strides': (1, 1), 'expand_ratio': 1, 'in_channels': stem_channels, 'out_channels': round_filters(16, width_coeff), 'se_ratio': 0.25},
      {'num_repeat': round(math.ceil(2 * depth_coeff)), 'kernel_size': (3, 3), 'strides': (2, 2), 'expand_ratio': 6, 'in_channels': round_filters(16, width_coeff), 'out_channels': round_filters(24, width_coeff), 'se_ratio': 0.25},
      {'num_repeat': round(math.ceil(2 * depth_coeff)), 'kernel_size': (5, 5), 'strides': (2, 2), 'expand_ratio': 6, 'in_channels': round_filters(24, width_coeff), 'out_channels': 40, 'se_ratio': 0.25},
      {'num_repeat': round(math.ceil(3 * depth_coeff)), 'kernel_size': (3, 3), 'strides': (2, 2), 'expand_ratio': 6, 'in_channels': 40, 'out_channels': 80, 'se_ratio': 0.25},
      {'num_repeat': round(math.ceil(3 * depth_coeff)), 'kernel_size': (5, 5), 'strides': (1, 1), 'expand_ratio': 6, 'in_channels': 80, 'out_channels': 112, 'se_ratio': 0.25},
      {'num_repeat': round(math.ceil(4 * depth_coeff)), 'kernel_size': (5, 5), 'strides': (2, 2), 'expand_ratio': 6, 'in_channels': 112, 'out_channels': 192, 'se_ratio': 0.25},
      {'num_repeat': round(math.ceil(1 * depth_coeff)), 'kernel_size': (3, 3), 'strides': (1, 1), 'expand_ratio': 6, 'in_channels': 192, 'out_channels': 320, 'se_ratio': 0.25},
    ]

    seed_index = 0
    for stage_idx, block_arg in enumerate(blocks_args):
      for repeat_idx in range(block_arg['num_repeat']):
        strides = block_arg['strides'] if repeat_idx == 0 else (1, 1)
        in_channels = block_arg['in_channels'] if repeat_idx == 0 else block_arg['out_channels']
        block_name = f'block{stage_idx + 1}{chr(ord("a") + repeat_idx)}'
        x = MBConv(
          expand_ratio=block_arg['expand_ratio'],
          kernel_size=block_arg['kernel_size'],
          strides=strides,
          in_channels=in_channels,
          out_channels=block_arg['out_channels'],
          se_ratio=block_arg['se_ratio'],
          drop_rate=0.0,  # You can adjust this if needed
          name=block_name
        )(x, train=train)

        if repeat_idx >= 1:
          seed_name = 'seed_generator' if seed_index == 0 else f'seed_generator_{seed_index}'
          x = DropPathState(name=seed_name)(x)
          seed_index += 1

    top_channels = 1280
    x = linen.Conv(top_channels, kernel_size=(1, 1), padding='SAME', use_bias=False, name='top_conv')(x)
    x = linen.BatchNorm(use_running_average=not train, name='top_bn')(x)
    x = linen.swish(x)

    return x

  def forward(self, img):
    weights = self.pretrained_weights()
    return self.apply(weights, img, train=False, mutable=False)
    
  def pretrained_weights(self):
    eff = keras.applications.EfficientNetB1(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    weights = eff.variables
    variables = transform_weights(weights)
    return variables


def format_path(weights):
    master_dict = {}
    for key in weights:
        master_dict[key] = {}
        for path in weights[key]:
            minor_node = master_dict[key]
            for folder in path.split('/'):
                minor_node[folder] = minor_node.get(folder, {})
                minor_node = minor_node[folder]

            for minor_key in weights[key][path]:
                minor_node[minor_key] = weights[key][path][minor_key]

    return regularize(master_dict)

def regularize(weights):

    def mapper(path, param):
        if param.ndim == 4 and 'dwconv' in ' '.join(x.key for x in path):
            return param.transpose((0, 1, 3, 2))
        else:
            return param

    return jax.tree.map_with_path(mapper, weights)

# To transform the weights
def transform_weights(tf_weights):
  flax_params = {}
  flax_batch_stats = {}

  for param in tf_weights:
    path = param.path
    if path.startswith('block'):
        path = path[:7] + '/' + path[8:]
    value = param.value.numpy()
    parts = path.split('/')
    if path.startswith('normalization'):
      if 'mean' in path:
        flax_params.setdefault('normalization', {})['mean'] = value
      elif 'variance' in path:
        flax_params.setdefault('normalization', {})['variance'] = value
      elif 'count' in path:
        flax_params.setdefault('normalization', {})['count'] = value
    elif path.startswith('seed_generator'):
      name = parts[0]
      flax_params.setdefault(name, {})['seed_generator_state'] = value
    elif 'bn' in parts[-2]:
      module = '/'.join(parts[:-1])
      if 'gamma' in path:
        flax_params.setdefault(module, {})['scale'] = value
      elif 'beta' in path:
        flax_params.setdefault(module, {})['bias'] = value
      elif 'moving_mean' in path:
        flax_batch_stats.setdefault(module, {})['mean'] = value
      elif 'moving_variance' in path:
        flax_batch_stats.setdefault(module, {})['var'] = value
    else:
      module = '/'.join(parts[:-1])
      key = parts[-1]
      flax_params.setdefault(module, {})[key] = value

  return format_path({'params': flax_params, 'batch_stats': flax_batch_stats})
