import torch.optim as optim

"List of hyperparams given for the model"

do_boltzman_exploration = True
memory_size = 1_000_000
min_rb_size = 50_000
sample_size = 32
lr = 0.0001
eps_max = 1.0
eps_min = 0.1
eps_decay = 0.999999  # Used as eps_decay ** num, for example will the eps after 1_000_000 steps with eps_decay = 0.999999 be 0,367879257. Will hit 0.1 around 2300000
discount_factor = 0.99
env_steps_before_train = 16
tgt_model_update = 5000
epochs_before_test = 1500

optimizer_function = optim.Adam
# optimizer_function = optim.RMSprop
