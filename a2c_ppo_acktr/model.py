import numpy as np
import torch
import torch.nn as nn

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import calculate_next_feature_map_size
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.utils import init_critic_layer
from a2c_ppo_acktr.utils import init_layer_in_actor


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base == 'Mnist':
            base = CNNMnist
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape, **base_kwargs)  # TODO Lia changed this, the original is the below commendted line
        #self.base = base(obs_shape[0], **base_kwargs)
        print("obs_shape[1]:", obs_shape[1])
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, obs_shape, recurrent=False, hidden_size=512, num_feature_maps=32, kernel_size=6):
        """ At first kernels were 8, 4, 3  now we try 6,4,3, this is for all growspace except mnist """

        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        assert len(obs_shape) == 3
        num_inputs, input_width, _ = obs_shape
        print("number of inputs:", num_inputs)

        self.cnn_layer1 = init_layer_in_actor(nn.Conv2d(num_inputs, num_feature_maps, kernel_size, stride=4, padding=0))
        self.cnn_layer2 = init_layer_in_actor(
            nn.Conv2d(num_feature_maps, num_feature_maps * 2, int(kernel_size / 2), stride=2, padding=0))
        self.cnn_layer3 = init_layer_in_actor(
            nn.Conv2d(num_feature_maps * 2, num_feature_maps, int(kernel_size / 2), stride=1, padding=0))

        feature_map_for_linear_layer = calculate_next_feature_map_size(
            [self.cnn_layer1, self.cnn_layer2, self.cnn_layer3], input_width
        )
        self.linear_layer = init_layer_in_actor(
            nn.Linear(num_feature_maps * feature_map_for_linear_layer ** 2, hidden_size))
        self.activation = nn.ReLU()

        self.critic_linear = init_critic_layer(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs / 255.0
        x = self.activation(self.cnn_layer1(x))
        x = self.activation(self.cnn_layer2(x))
        x = self.activation(self.cnn_layer3(x))
        x = x.view(-1, x.shape[1:].numel())
        x = self.activation(self.linear_layer(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class CNNMnist(NNBase):
    def __init__(self, obs_shape, recurrent=False, hidden_size=512, num_feature_maps=32, kernel_size=3):
        """ Start with kernel 3 for mnist challenges """

        super(CNNMnist, self).__init__(recurrent, hidden_size, hidden_size)

        assert len(obs_shape) == 3
        num_inputs, input_width, _ = obs_shape
        print("number of inputs:", num_inputs)

        self.cnn_layer1 = init_layer_in_actor(
            nn.Conv2d(num_inputs, num_feature_maps, kernel_size, stride=4, padding=0))
        self.cnn_layer2 = init_layer_in_actor(
            nn.Conv2d(num_feature_maps, num_feature_maps * 2, int(kernel_size / 2), stride=2, padding=0))
        self.cnn_layer3 = init_layer_in_actor(
            nn.Conv2d(num_feature_maps * 2, num_feature_maps, int(kernel_size / 2), stride=1, padding=0))

        feature_map_for_linear_layer = calculate_next_feature_map_size(
            [self.cnn_layer1, self.cnn_layer2, self.cnn_layer3], input_width
        )
        self.linear_layer = init_layer_in_actor(
            nn.Linear(num_feature_maps * feature_map_for_linear_layer ** 2, hidden_size))
        self.activation = nn.ReLU()

        self.critic_linear = init_critic_layer(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs / 255.0
        x = self.activation(self.cnn_layer1(x))
        x = self.activation(self.cnn_layer2(x))
        x = self.activation(self.cnn_layer3(x))
        x = x.view(-1, x.shape[1:].numel())
        x = self.activation(self.linear_layer(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
