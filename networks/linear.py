import torch
import numpy as np
import typing as t
import torch.nn as nn
import torch.nn.functional as F

from torch import FloatTensor
from torch.distributions.normal import Normal

from networks.core import mlp


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: t.List[int],
        log_min_std: float = -20,
        log_max_std: float = 2,
        act_limit: float = 10
    ):
        super(Actor, self).__init__()
        self.layers = mlp([obs_dim] + hidden_sizes)
        self.act_dim = act_dim
        self.mu_layer = nn.Linear(hidden_sizes[-1], self.act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], self.act_dim)
        self.log_min_std = log_min_std
        self.log_max_std = log_max_std
        self.act_limit = act_limit

    def forward(self, x: FloatTensor, deterministic: bool = False, with_logprob: bool = True):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clip(log_std, self.log_min_std, self.log_max_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)

        prob = mu
        if not deterministic:
            prob = pi_distribution.rsample()
        pi_action = torch.tanh(prob) * self.act_limit

        logprob = None
        if with_logprob:
            logprob = pi_distribution.log_prob(prob).sum(dim=-1)
            logprob -= (2 * torch.as_tensor(np.log(2)) - prob - F.softplus(-2*prob)).sum(dim=-1)

        return (pi_action, logprob)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: t.List[int]):
        super(Critic, self).__init__()
        self.layers = mlp([obs_dim + act_dim] + hidden_sizes + [1])

    def forward(self, x: FloatTensor) -> FloatTensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)

        x = self.layers[-1](x)
        x = torch.squeeze(x, -1)
        return x


class DoubleCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: t.List[int]):
        super(DoubleCritic, self).__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes)
        self.q2 = Critic(obs_dim, act_dim, hidden_sizes)

    def forward(self, x: FloatTensor) -> t.Tuple[FloatTensor, FloatTensor]:
        return self.q1(x), self.q2(x)
