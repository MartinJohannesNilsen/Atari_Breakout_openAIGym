"List of hyperparameters given for the machine learning model and agent"
from utils import Boltzmann, EpsilonGreedy
import torch.optim as optim

memory_size = 1_000_000
min_rb_size = 50_000
sample_size = 32
lr = 0.0001
eps_min = 0.1
eps_decay = 0.999999  # Used as eps_decay ** num, for example will the eps after 1_000_000 steps with eps_decay = 0.999999 be 0,367879257. Will hit 0.1 around 2300000
discount_factor = 0.99
env_steps_before_train = 16
epochs_before_tgt_model_update = 5000
epochs_before_test = 1500

optimizer_function = optim.Adam
# optimizer_function = optim.RMSprop

exploration_method = Boltzmann
# exploration_method = EpsilonGreedy


"""
Første gang kjører den min_rb_size ganger før den går videre, for å fylle replaybuffer
16 steg før den begynner trene
1500 epoker før test
og 5000 epoker før den oppdaterer target model
"""
