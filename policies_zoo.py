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

    envs = make_vec_env(config.env_name, n_envs=config.num_processes)

    model = PPO(
        "CnnPolicy", envs, verbose=1, tensorboard_log="./runs/", clip_range=config.clip_param, n_steps=2000,
        learning_rate=config.lr, gamma=config.gamma, gae_lambda=config.gae_lambda, ent_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm, vf_coef=config.value_loss_coef, batch_size=config.num_mini_batch
    )
    model.learn(total_timesteps=config.num_steps, log_interval=1, callback=WandbStableBaselines3Callback())
    model.save(f"{config.env_name}_stable_baselines_ppo")


if __name__ == "__main__":
    ppo_stable_baselines_training()
