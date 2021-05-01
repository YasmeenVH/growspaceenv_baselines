import torch
import numpy as np
import experiment_buddy
""" 
old values of growspace 

FIRST_BRANCH_HEIGHT = .42
BRANCH_THICCNESS = 0.015
BRANCH_LENGTH = 1 / 9
MAX_BRANCHING = 10
LIGHT_WIDTH = .25
LIGHT_DIF = 250
"""

algo = "ppo"  # no change
gail = False   # not important
gail_experts_dir = './gail_experts'  # not important
gail_batch_size = 128                # not important
gail_epoch = 5                       # not important
#lr = 2.5e-4
lr = 0.006697
#eps = 1e-5
eps = 0.03068
alpha = 0.99                         # for a2c not ppo
#gamma = 0.99
gamma = 0.8964
#use_gae = True
use_gae = False
#gae_lambda = 0.95
gae_lambda = 0.3429
#entropy_coef = 0.01
entropy_coef = 0.1316
#value_loss_coef = 0.5
value_loss_coef = 0.3638
#max_grad_norm = 0.5
max_grad_norm = 0.3406
seed = np.random.randint(1,80, size = 1)   # didnt change
cuda_deterministic = False
num_processes = 1
#num_steps = 2500
num_steps = 4240
custom_gym = "growspace"
#ppo_epoch = 4
ppo_epoch = 15
#num_mini_batch = 32
num_mini_batch = 25
#clip_param = 0.1
clip_param = 0.3758
log_interval = 10  # amount of times we save to wandb
save_interval = 100 # amount of times we save internal
eval_interval = None
num_env_steps = 1e6  # no change
env_name = "GrowSpaceEnv-Control-v0"#"GrowSpaceSpotlight-Mnist4-v0"
log_dir = "/tmp/gym/"
save_dir = "./trained_models/"
use_proper_time_limits = False
recurrent_policy = False
use_linear_lr_decay = True
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
optimizer = "adam"
momentum = 0.95


experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "mila",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "growspace"}
)
