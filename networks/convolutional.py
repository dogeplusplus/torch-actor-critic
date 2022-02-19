import torch
import typing as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch import FloatTensor
from torch.distributions.normal import Normal

from networks.core import mlp
from environments.wall_runner import MultiObservation


def calculate_size(
    image_shape: t.Tuple[int, int, int],
    filters: t.List[int],
    kernel_sizes: t.List[int],
    strides: t.List[int],
) -> int:
    image_shape = list(image_shape)

    for f, k, s in zip(filters, kernel_sizes, strides):
        image_shape[0] = f
        image_shape[1] = int(np.floor((image_shape[1] - k) / s + 1))
        image_shape[2] = int(np.floor((image_shape[2] - k) / s + 1))

    return np.prod(image_shape)


def simple_cnn(
    input_shape: t.Tuple[int, int, int],
    filters: t.List[int] = [32, 64, 64],
    kernel_sizes: t.List[int] = [8, 4, 3],
    strides: t.List[int] = [4, 2, 1],
    activation: nn.Module = nn.ReLU,
    dense_size: int = 512,
) -> nn.Module:
    channels, _, _ = input_shape

    model = nn.Sequential()
    sizes = [channels] + filters
    for i, _ in enumerate(sizes[:-1]):
        model.add_module(f"conv_{i}", nn.Conv2d(sizes[i], sizes[i+1], kernel_sizes[i], strides[i]))
        model.add_module(f"relu_{i}", activation())

    conv_shape = calculate_size(input_shape, filters, kernel_sizes, strides)
    model.add_module("flatten", nn.Flatten())
    model.add_module("linear", nn.Linear(conv_shape, dense_size))
    model.add_module("final", nn.Linear(dense_size, 1))

    return model


class VisualActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        vis_dim: t.Tuple[int, int, int],
        hidden_sizes: t.List[int] = [256, 256],
        log_min_std: float = -20,
        log_max_std: float = 2,
        act_limit: float = 10,
        filters: t.List[int] = [32, 64, 64],
        kernel_sizes: t.List[int] = [8, 4, 3],
        strides: t.List[int] = [4, 2, 1],
    ):
        super(VisualActor, self).__init__()
        self.layers = mlp([obs_dim] + hidden_sizes)

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.vis_dim = vis_dim

        self.visual_network = simple_cnn(vis_dim, filters, kernel_sizes, strides)

        # Add 1 for the visual information
        self.mu_layer = nn.Linear(hidden_sizes[-1] + 1, self.act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1] + 1, self.act_dim)
        self.log_min_std = log_min_std
        self.log_max_std = log_max_std
        self.act_limit = act_limit

    def forward(
        self,
        x: MultiObservation,
        deterministic: bool = False,
        with_logprob: bool = True
    ) -> FloatTensor:
        image = x.camera
        image = image.view((-1,) + self.vis_dim)
        x = x.features
        x = x.view((-1, self.obs_dim))

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        conv_output = self.visual_network(image)
        x = torch.cat([x, conv_output], dim=1)
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

        return (torch.squeeze(pi_action), torch.squeeze(logprob))


class VisualCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        vis_dim: t.Tuple[int, int, int],
        hidden_sizes: t.List[int] = [256, 256],
        filters: t.List[int] = [32, 64, 64],
        kernel_sizes: t.List[int] = [8, 4, 3],
        strides: t.List[int] = [4, 2, 1],
    ):
        super(VisualCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.vis_dim = vis_dim

        # Extra neuron in penultimate fully connected layer for visual information
        self.layers = mlp([obs_dim + act_dim] + hidden_sizes + [1])
        self.final = nn.Linear(2, 1)
        self.visual_network = simple_cnn(vis_dim, filters, kernel_sizes, strides)

    def forward(self, x: MultiObservation) -> FloatTensor:
        image = x.camera
        image = image.view((-1,) + self.vis_dim)
        conv_output = self.visual_network(image)

        x = x.features
        x = x.view((-1, self.obs_dim + self.act_dim))
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

        x = torch.cat([x, conv_output], dim=1)
        x = self.final(x)
        x = torch.squeeze(x, -1)

        return x


class VisualDoubleCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        vis_dim: t.Tuple[int, int, int],
        hidden_sizes: t.List[int] = [256, 256],
        filters: t.List[int] = [32, 64, 64],
        kernel_sizes: t.List[int] = [8, 4, 3],
        strides: t.List[int] = [4, 2, 1],
    ):
        super(VisualDoubleCritic, self).__init__()
        self.q1 = VisualCritic(obs_dim, act_dim, vis_dim, hidden_sizes, filters, kernel_sizes, strides)
        self.q2 = VisualCritic(obs_dim, act_dim, vis_dim, hidden_sizes, filters, kernel_sizes, strides)

    def forward(self, x: MultiObservation) -> t.Tuple[FloatTensor, FloatTensor]:
        return self.q1(x), self.q2(x)
