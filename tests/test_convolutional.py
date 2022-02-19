import torch

from networks.convolutional import VisualActor, VisualCritic
from environments.wall_runner import MultiObservation


def test_visual_actor():
    obs_dim = 20
    act_dim = 10
    vis_dim = (3, 64, 64)
    model = VisualActor(obs_dim, act_dim, vis_dim)

    features = torch.rand((obs_dim,))
    camera = torch.rand(vis_dim)
    x = MultiObservation(features, camera)
    pi, log_prob = model(x)

    assert list(pi.size()) == [act_dim]
    assert list(log_prob.size()) == []


def test_visual_double_critics():
    obs_dim = 20
    act_dim = 10
    vis_dim = (3, 64, 64)
    model = VisualCritic(obs_dim, act_dim, vis_dim)

    state_action = torch.rand((2, obs_dim + act_dim))
    camera = torch.rand((2,) + vis_dim)
    x = MultiObservation(state_action, camera)
    value = model(x)
    assert list(value.size()) == [2]

    state_action = torch.rand((obs_dim + act_dim,))
    camera = torch.rand(vis_dim)
    x = MultiObservation(state_action, camera)
    value = model(x)
    assert list(value.size()) == [1]
