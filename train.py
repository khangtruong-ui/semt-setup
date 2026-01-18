print('===== TRAINING SCRIPT =====')

import os

from model import MeshedFastCaption
from train_util import *
from data_utils import get_train_set

print('===== CRAFTING DATASETS =====')
ds, ds_length = get_train_set()

print('====== CRAFTING MODELS ======')
model = MeshedFastCaption()
state = create_train_state(model)

print('===== TRAINING ======')
state = train_loop(model, state, ds, ds_length, 60)
