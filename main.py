#try:
#    from comet_ml import Experiment
#    comet_loaded = True
#except ImportError:
#    comet_loaded = False
import copy
import glob
import os
import sys
import time
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**84)
import cv2
from comet_ml import Experiment
import config

def main():

    #if comet_loaded:
       # experiment = Experiment(
            #api_key="WRmA8ms9A78K85fLxcv8Nsld9",
            #project_name="growspace-tests",
            #workspace="yasmeenvh")
        #experiment.set_name(config.comet)
        #for key, value in vars(config).items():
            #experiment.log_parameter(key, value)
    #else:
        #experiment = None
    # wandb.init(settings=wandb.Settings(start_method="fork"),project='ppo', entity='growspace')
    #wandb.config()
    wandb.run = config.tensorboard.run

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.cuda and torch.cuda.is_available() and config.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(config.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if config.cuda else "cpu")

    envs = make_vec_envs(config.env_name, config.seed, config.num_processes,
                         config.gamma, config.log_dir, device, False, config.custom_gym)

    if "Mnist" in config.env_name:
        base = 'Mnist'
    else:
        base = None

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base,
        base_kwargs={'recurrent': config.recurrent_policy})
    actor_critic.to(device)
    evaluate(actor_critic, None, config.env_name, config.seed, config.num_processes, eval_log_dir, device, config.custom_gym)

    if config.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            alpha=config.alpha,
            max_grad_norm=config.max_grad_norm)
    elif config.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            config.clip_param,
            config.ppo_epoch,
            config.num_mini_batch,
            config.value_loss_coef,
            config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm)
    elif config.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, config.value_loss_coef, config.entropy_coef, acktr=True)

    if config.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            config.gail_experts_dir, "trajs_{}.pt".format(
                config.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > config.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=config.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(config.num_steps, config.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = [] #deque(maxlen=50)
    episode_length = [] #deque(maxlen=50)
    episode_branches = [] #deque(maxlen=50)
    episode_branch1 = [] #deque(maxlen=50)
    episode_branch2 = [] #deque(maxlen=50)
    episode_light_width = []#deque(maxlen=50)
    episode_light_move = [] #deque(maxlen=50)
    episode_success = [] #deque(maxlen=50)
    #episode_light_move = deque(maxlen=10)
    #new_branches = []
    #episode_success_rate = [] #deque(maxlen=100)
    episode_total = 0

    start = time.time()
    num_updates = int(
        config.num_env_steps) // config.num_steps // config.num_processes
    #print("what are the num_updates",num_updates)
    x = 0
    #s = 0
    #av_ep_move = []
    for j in range(num_updates):

        if config.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if config.algo == "acktr" else config.lr)
        #new_branches = []
        for step in range(config.num_steps):
            #if s == 0:
                #light_move_ep = []
            #s +=1
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            #misc = {"tips": tips, "target": self.target, "light": self.x1_light, "light width": LIGHT_WIDTH, "step": self.steps. "new_branches": self.new_branches}

            for info in infos:
                #print("what is info:",info.keys())
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_length.append(info['episode']['l'])
                    #print("ep rewards:", episode_rewards)

                if 'new_branches' in info.keys():
                    episode_branches.append(info['new_branches'])
                    #print("what is in new branches", info['new_branches'])
                    # print("type of data:", type(info['new_branches']))

                    #print("what is new branches", new_branches)
                if 'new_b1' in info.keys():
                    episode_branch1.append(info['new_b1'])

                if 'new_b2' in info.keys():
                    episode_branch2.append(info['new_b2'])

                if 'light_width' in info.keys():
                    episode_light_width.append(info['light_width'])
                    #print("what is new branches", new_branches)

                if 'light_move' in info.keys():
                    episode_light_move.append(info['light_move'])
                    #light_move_ep.append(info['light_move'])
                    #print("what is new branches", new_branches)

                if 'success' in info.keys():
                    episode_success.append(info['success'])
                    # print("what is new branches", new_branches)

                if j == x:
                    if 'img' in info.keys():
                        img = info['img']
                        path = './hittiyas/growspaceenv_braselines/scripts/imgs/'
                        cv2.imwrite(os.path.join(path, 'step' + str(step) + '.png'), img)
                    x += 1000

            #if s == 50:
                #av = np.mean(light_move_ep)
                #av_ep_move.append(av)
                #s = 0

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        #print("before")
        #episode_branches.append(np.asarray([[np.mean(new_branches)]]))
        #print("after")
        #print(episode_branches)
        if config.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = config.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(config.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], config.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, config.use_gae, config.gamma,
                                 config.gae_lambda, config.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()


        # save for every interval-th episode or for the last epoch
        if (j % config.save_interval == 0
                or j == num_updates - 1) and config.save_dir != "":
            save_path = os.path.join(config.save_dir, config.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, config.env_name + ".pt"))

        if j % config.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config.num_processes * config.num_steps
            end = time.time()
            wandb.log({"Reward Min": np.min(episode_rewards)}, step=total_num_steps)
            wandb.log({"Episode Reward": episode_rewards}, step=total_num_steps)
            wandb.log({"Summed Reward": np.sum(episode_rewards)}, step=total_num_steps)
            wandb.log({"Reward Mean": np.mean(episode_rewards)}, step=total_num_steps)
            wandb.log({"Reward Max": np.max(episode_rewards)}, step=total_num_steps)
            wandb.log({"Number of Mean New Branches": np.mean(episode_branches)}, step=total_num_steps)
            wandb.log({"Number of Max New Branches": np.max(episode_branches)}, step=total_num_steps)
            wandb.log({"Number of Min New Branches": np.min(episode_branches)}, step=total_num_steps)
            wandb.log({"Number of Mean New Branches of Plant 1": np.mean(episode_branch1)}, step = total_num_steps)
            wandb.log({"Number of Mean New Branches of Plant 2": np.mean(episode_branch2)}, step=total_num_steps)
            wandb.log({"Number of Total Displacement of Light": np.sum(episode_light_move)}, step=total_num_steps)
            wandb.log({"Mean Light Displacement": episode_light_move}, step=total_num_steps)
            wandb.log({"Mean Light Width": episode_light_width}, step=total_num_steps)
            wandb.log({"Number of Steps in Episode with Tree is as close as possible": np.sum(episode_success)},step=total_num_steps)


            # if experiment is not None:
            #     experiment.log_metric(
            #         "Reward Mean",
            #         np.mean(episode_rewards),
            #         step=total_num_steps)
            #     experiment.log_metric(
            #         "Reward Min", np.min(episode_rewards), step=total_num_steps)
            #     experiment.log_metric(
            #         "Reward Max", np.max(episode_rewards), step=total_num_steps)
            #     experiment.log_metric(
            #         "Number of Mean New Branches", np.mean(episode_branches), step=total_num_steps)
            #     experiment.log_metric(
            #         "Number of Total New Branches", np.sum(episode_branches), step=total_num_steps)
            #     experiment.log_metric(
            #         "Number of Min New Branches", np.min(episode_branches), step=total_num_steps)
            #     experiment.log_metric(
            #         "Number of Max New Branches", np.max(episode_branches), step=total_num_steps)
            #
            #     experiment.log_metric(
            #         "Number of Mean New Branches of Plant 1", np.mean(episode_branch1), step=total_num_steps)
            #     experiment.log_metric(
            #         "Number of Mean New Branches of Plant 2", np.mean(episode_branch2), step=total_num_steps)
            #
            #     experiment.log_metric(
            #         "Number of Total Displacement of Light", np.sum(episode_light_move), step=total_num_steps)
            #     experiment.log_metric(
            #         "Mean Displacement of Light", np.mean(episode_light_move), step=total_num_steps)
            #     experiment.log_metric(
            #         "Mean Light Width", np.mean(episode_light_width), step=total_num_steps)
            #     experiment.log_metric(
            #         "Number of Steps in Episode with Tree is as close as possible", np.sum(episode_success), step=total_num_steps)
            #     experiment.log_metric(
            #         "Episode Length Mean ",
            #         np.mean(episode_length),
            #         step=total_num_steps)
            #     experiment.log_metric(
            #         "Episode Length Min",
            #         np.min(episode_length),
            #         step=total_num_steps)
            #     experiment.log_metric(
            #         "Episode Length Max",
            #         np.max(episode_length),
            #         step=total_num_steps)

            #print("Number of mean branches", np.mean(episode_branches))
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            episode_rewards.clear()
            episode_length.clear()
            episode_branches.clear()
            episode_branch2.clear()
            episode_branch1.clear()
            episode_light_move.clear()
            episode_light_width.clear()
            episode_success.clear()


        if (config.eval_interval is not None and len(episode_rewards) > 1
                and j % config.eval_interval == 0):
            ob_rms = getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            evaluate(actor_critic, ob_rms, config.env_name, config.seed, config.num_processes, eval_log_dir, device,  config.custom_gym)

    ob_rms = getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
    evaluate(actor_critic, ob_rms, config.env_name, config.seed, config.num_processes, eval_log_dir, device, config.custom_gym)


if __name__ == "__main__":
    main()
