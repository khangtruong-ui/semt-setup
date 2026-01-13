from model import MeshedFastCaption
from train_util import *
from data_utils import get_train_set

ds = get_train_set()

model = MeshedFastCaption()
state = create_train_state(model)
state = train_loop(state, ds, 1)
