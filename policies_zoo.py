import config

import os

import gym
from gym.envs.registration import register
# from stable_baselines import PPO2
# from stable_baselines.common.policies import CnnPolicy
import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
from gym.spaces.discrete import Discrete

import wandb

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10)
#
# obs = env.reset()
# for i in range(10):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()
#
# env.close()

register(
    id=f'GrowSpaceEnv-Continuous-v0',
    entry_point='growspace.env:GrowSpaceContinuous',
    max_episode_steps=50,
)


def ppo2_stable_baselines_training():
    wandb.run = config.tensorboard.run
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    envs = gym.make(config.env_name)

    model = PPO2(
        CnnPolicy, envs, verbose=1, tensorboard_log="./ppo-stable-tensorboard-plant/", cliprange=0.2, n_steps=2500
    )
    model.learn(total_timesteps=1000000, log_interval=10)
    model.save("stable_baselines_ppo2_continuous_plant")


if __name__ == "__main__":
    ppo2_stable_baselines_training()
