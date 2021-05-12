import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb


class WandbStableBaselines3Callback(BaseCallback):

    def __init__(self, verbose=0, enable_popping_up_window=False, model_name=None, env_name=None):
        super(WandbStableBaselines3Callback, self).__init__(verbose)
        self.enable_popping_up_window = enable_popping_up_window
        self.model_name = model_name
        self.env_name = env_name

        self.frames_for_gif = []

        self.episode_rewards = []
        self.episode_length = []
        self.episode_branches = []
        self.episode_branch1 = []
        self.episode_branch2 = []
        self.episode_light_width = []
        self.episode_light_move = []
        self.episode_success = []
        self.episode_plantpixel = []

        self.count_time_steps = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        infos = self.locals["infos"]
        for info in infos:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])
                self.episode_length.append(info['episode']['l'])
                wandb.log({"Episode_Reward": info['episode']['r']}, step=self.count_time_steps)
                self.count_time_steps += 1

            if 'new_branches' in info.keys():
                self.episode_branches.append(info['new_branches'])

            if 'new_b1' in info.keys():
                self.episode_branch1.append(info['new_b1'])

            if 'new_b2' in info.keys():
                self.episode_branch2.append(info['new_b2'])

            if 'light_width' in info.keys():
                self.episode_light_width.append(info['light_width'])

            if 'light_move' in info.keys():
                self.episode_light_move.append(info['light_move'])

            if 'success' in info.keys():
                self.episode_success.append(info['success'])

            if 'plant_pixel' in info.keys():
                self.episode_plantpixel.append(info['plant_pixel'])

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        wandb.log({"Reward Min": np.min(self.episode_rewards)}, step=self.count_time_steps)
        wandb.log({"Summed Reward": np.sum(self.episode_rewards)}, step=self.count_time_steps)
        wandb.log({"Reward Mean": np.mean(self.episode_rewards)}, step=self.count_time_steps)
        wandb.log({"Reward Max": np.max(self.episode_rewards)}, step=self.count_time_steps)
        wandb.log({"Number of Mean New Branches": np.mean(self.episode_branches)}, step=self.count_time_steps)
        wandb.log({"Number of Max New Branches": np.max(self.episode_branches)}, step=self.count_time_steps)
        wandb.log({"Number of Min New Branches": np.min(self.episode_branches)}, step=self.count_time_steps)
        wandb.log({"Number of Mean New Branches of Plant 1": np.mean(self.episode_branch1)}, step=self.count_time_steps)
        wandb.log({"Number of Mean New Branches of Plant 2": np.mean(self.episode_branch2)}, step=self.count_time_steps)
        wandb.log({"Number of Total Displacement of Light": np.sum(self.episode_light_move)},
                  step=self.count_time_steps)
        wandb.log({"Mean Light Displacement": self.episode_light_move}, step=self.count_time_steps)
        wandb.log({"Mean Light Width": self.episode_light_width}, step=self.count_time_steps)
        wandb.log({"Number of Steps in Episode with Tree is as close as possible": np.sum(self.episode_success)},
                  step=self.count_time_steps)
        wandb.log({"Displacement of Light Position": wandb.Histogram(self.episode_light_move)},
                  step=self.count_time_steps)
        wandb.log({"Displacement of Beam Width": wandb.Histogram(self.episode_light_width)}, step=self.count_time_steps)
        wandb.log({"Mean Plant Pixel": np.mean(self.episode_plantpixel)}, step=self.count_time_steps)
        wandb.log({"Summed Plant Pixel": np.sum(self.episode_plantpixel)}, step=self.count_time_steps)
        wandb.log({"Plant Pixel Histogram": wandb.Histogram(self.episode_plantpixel)}, step=self.count_time_steps)

        self.episode_rewards.clear()
        self.episode_length.clear()
        self.episode_branches.clear()
        self.episode_branch2.clear()
        self.episode_branch1.clear()
        self.episode_light_move.clear()
        self.episode_light_width.clear()
        self.episode_success.clear()
        self.episode_plantpixel.clear()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass



