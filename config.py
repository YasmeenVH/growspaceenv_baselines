import torch

import experiment_buddy

FIRST_BRANCH_HEIGHT = .42
BRANCH_THICCNESS = 0.015
BRANCH_LENGTH = 1 / 9
MAX_BRANCHING = 10
LIGHT_WIDTH = .25
LIGHT_DIF = 250

lr = 0.07827
eps = 0.04517
gamma = 0.1388
use_gae = True
gae_lambda = 0.3265
entropy_coef = 0.308
value_loss_coef = 0.2915
max_grad_norm = 0.4387
num_steps = 3285
optimizer= "adam"
ppo_epoch = 18
num_mini_batch = 23
clip_param =0.09442
use_linear_lr_decay = True


algo = "ppo"
gail = False
gail_experts_dir = './gail_experts'
gail_batch_size = 128
gail_epoch = 5
alpha = 0.99
seed = 1
cuda_deterministic = False
num_processes = 1
custom_gym = "growspace"
log_interval = 10
save_interval = 100
eval_interval = None
num_env_steps = 1e6
env_name = "GrowSpaceEnv-Control-v0"
# env_name = "GrowSpaceEnv-Hierarchy-v0"
# env_name = "GrowSpaceSpotlight-MnistMix-v0"
# env_name = "GrowSpaceEnv-Fairness-v0"
log_dir = "/tmp/gym/"
save_dir = "./trained_models/"
use_proper_time_limits = False
recurrent_policy = False
use_linear_lr_decay = True
no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
optimizer = "adam"
momentum = 0.9  # if sgd is used

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "mila",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "growspace"}
)
