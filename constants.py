"List of defined hyperparameters and methods used for model optimization and agent exploration vs exploitation"
from utils import Boltzmann, EpsilonGreedy, FrameStackingAndResizingEnv, NoFireInActionSpaceEnv
import torch.optim as optim

"Hyperparameters"
memory_size = 1_000_000
min_rb_size = 50_000
sample_size = 32
lr = 0.0001
discount_factor = 0.99
env_steps_before_train = 16
epochs_before_tgt_model_update = 5000
epochs_before_test = 1500
episode_max_steps = 10_000

# eps_min = 0.05  # Epsilon Greedy
eps_min = 0.1  # Boltzmann

eps_decay = 0.999999  # Used as eps_decay ** num, for example will the eps after 1 000 000 steps with eps_decay = 0.999999 be 0,367879257. Will hit 0.1 around 2300000
# eps_decay = 0.9999  # Faster decrease, if you want to train again using model checkpoint


"Other parameters"
optimizer_function = optim.RMSprop
# optimizer_function = optim.Adam

# exploration_method = EpsilonGreedy
exploration_method = Boltzmann

# env_type = FrameStackingAndResizingEnv
env_type = NoFireInActionSpaceEnv
