import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(in_dim, out_dim, hidden_size, out_gain):
    layers = [nn.Linear(in_dim, hidden_size), nn.Tanh(),
              nn.Linear(hidden_size, hidden_size), nn.Tanh(),
              nn.Linear(hidden_size, out_dim)]
    # orthogonal init is the usual choice for PPO; the last layer gets its own gain
    for layer in layers:
        if isinstance(layer, nn.Linear):
            gain = out_gain if layer is layers[-1] else np.sqrt(2)
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    # every boid uses this same network
    # actor gives the mean of a Gaussian, log_std is one learned number per action dim (not per state)

    def __init__(self, obs_dim, act_dim, hidden_size=64, init_log_std=0.0):
        super().__init__()
        # tiny output gain so the untrained policy starts with near-zero mean actions
        self.actor = mlp(obs_dim, act_dim, hidden_size, out_gain=0.01)
        self.critic = mlp(obs_dim, 1, hidden_size, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def dist(self, obs):
        return Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        dist = self.dist(obs)
        action = dist.mean if deterministic else dist.sample()
        # the two action dims are independent, so the log probs just add up
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate(self, obs, actions):
        dist = self.dist(obs)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), self.value(obs)
