import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import FloatTensor
from typing import List, Tuple


def mlp(neurons: List[int]) -> nn.ModuleList:
    layers = nn.ModuleList([
        nn.Linear(x, y) for x, y in zip(neurons[:-1], neurons[1:])
    ])
    return layers


def softplus(x, beta: float = 1, threshold: float = 20) -> np.ndarray:
    softp = 1.0 / beta * np.log(1 + np.exp(beta * x))
    return np.where(beta * x > threshold, x, softp)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int],
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

        prob = mu
        if not deterministic:
            prob += std * torch.distributions.normal()

        logprob = None
        if with_logprob:
            logprob = torch.sum(torch.distributions.Normal(mu, std).log_prob(prob), dim=-1)
            logprob -= 2 * torch.sum(torch.tensor(np.log(2)) - prob - softplus(-2*prob), dim=-1)

        pi_action = torch.tanh(prob) * self.act_limit

        return (pi_action, logprob)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super(Critic, self).__init__()
        self.layers = mlp([obs_dim + act_dim] + hidden_sizes + [1])

    def forward(self, x: FloatTensor) -> FloatTensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)

        x = self.layers[-1](x)
        return x


class DoubleCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super(DoubleCritic, self).__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden_sizes)
        self.q2 = Critic(obs_dim, act_dim, hidden_sizes)

    def forward(self, x: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        return self.q1(x), self.q2(x)

