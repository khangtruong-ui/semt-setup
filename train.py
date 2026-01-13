import os

os.environ['DATASET'] = './ds'

from model import MeshedFastCaption
from train_util import *
from data_utils import get_train_set


model = MeshedFastCaption()
state = create_train_state(model)


ds = get_train_set()
state = train_loop(state, ds, 1)
