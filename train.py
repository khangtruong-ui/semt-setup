from model import MeshedFastCaption
from train_util import *


model = MeshedFastCaption()
state = create_train_state(model)
state = train_loop(state)
