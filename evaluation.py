import array2gif
import numpy as np
import torch

import config
import wandb
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir, device, custom_gym, gif=False):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes, None, eval_log_dir, device, True, custom_gym)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    images = []
    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            images.append(obs[0, -3:, :].squeeze().cpu().numpy())
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    images.append(obs[0, -3:, :].squeeze().cpu().numpy())
    eval_envs.close()
    if gif:
        array2gif.write_gif(images, 'replay.gif', fps=4)
        config.tensorboard.run.log({"video": wandb.Video('replay.gif', fps=4, format="gif")}, commit=True)
        config.tensorboard.run.history._flush()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(len(eval_episode_rewards), np.mean(eval_episode_rewards)))
