# noinspection PyUnresolvedReferences
import growspace
import torch
import torch.backends.cudnn
import torch.utils.data
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import config
from callbacks import WandbStableBaselines3Callback


def ppo_stable_baselines_training():
    wandb.run = config.tensorboard.run
    wandb.tensorboard.patch(save=False, tensorboardX=True)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    envs = make_vec_env(config.env_name, n_envs=4)

    model = PPO("CnnPolicy", envs, verbose=1, tensorboard_log="./runs/", clip_range=0.2, n_steps=50)
    model.learn(total_timesteps=500, log_interval=1, callback=WandbStableBaselines3Callback())
    model.save(f"{config.env_name}_stable_baselines_ppo")


if __name__ == "__main__":
    ppo_stable_baselines_training()
