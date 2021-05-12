import logging
import os
import random

import numpy as np
import torch

import wandb
from matplotlib import animation
import matplotlib.pyplot as plt


def save_frames_as_gif(frames, path='./', filename='growspace_with_ppo.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def render_to_gif():
    def save_frames_as_gif(frames, path='./', filename='growspace_with_trpo.gif'):
        # Mess with this to change frame size
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        anim.save(path + filename, writer='imagemagick', fps=60)

    env = gym.make('GrowSpaceEnv-Control-v0')
    model = TRPO(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=2500)
    # model.save("trpo_cartpole")

    # del model  # remove to demonstrate saving and loading

    model = TRPO.load("trpo_cartpole")

    frames = []
    obs = env.reset()
    for _ in range(150):
        # while True:
        frames.append(env.render(mode="rgb_array"))

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # if done:
        #     break
        # env.render()

    env.close()
    save_frames_as_gif(frames)


def init_wandb_run(project_name, env_name, model_name, run_name, dir="/publicwork/students/buzatu/"):
    wandb.init(project=project_name, reinit=True, dir=dir, monitor_gym=True)
    wandb.run.name = f"DELETE{env_name}/{model_name}"
    logging.info(f"init run name: {run_name}")


def delete_wandb_run(run_name):
    api = wandb.Api()
    run = api.run(run_name)
    run.delete()
    logging.info(f"run {run_name} had been deleted with success")


def save_model_weights(model, model_name, env_id, policy, seed, path="/publicwork/students/buzatu/weights"):
    save_model_path = f"{path}/{model_name}"
    os.makedirs(save_model_path, exist_ok=True)
    model.save(f"{save_model_path}/{model_name}_{env_id}_{policy}_{seed}")


def set_global_seed(seed=123):
    # Tensorflow
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # NumPy
    np.random.seed(seed)

    # Python
    random.seed(seed)


def watch_agent():
    global action, info
    # After training, watch our agent walk
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


def example_save_model_properly():
    # Custom MLP policy of three layers of size 128 each
    class CustomPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomPolicy, self).__init__(*args, **kwargs,
                                               layers=[128, 128, 128],
                                               feature_extraction="mlp")

    # Create and wrap the environment
    env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: env])

    model = A2C(CustomPolicy, env, verbose=1)
    # Train the agent
    model.learn(total_timesteps=1000)
    model.save("test_save")

    del model

    model = A2C.load("test_save", env=env, policy=CustomPolicy)
