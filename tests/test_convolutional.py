import torch

from networks.convolutional import VisualActor, VisualCritic
from environments.wall_runner import MultiObservation


def test_visual_actor():
    obs_dim = 20
    act_dim = 10
    vis_dim = (3, 64, 64)
    model = VisualActor(obs_dim, act_dim, vis_dim)

    features = torch.rand((obs_dim,))
    frame = torch.rand(vis_dim)
    x = MultiObservation(features, frame)
    pi, log_prob = model(x)

    assert list(pi.size()) == [act_dim]
    assert list(log_prob.size()) == []


def test_visual_double_critics():
    obs_dim = 20
    act_dim = 10
    vis_dim = (3, 64, 64)
    model = VisualCritic(obs_dim, act_dim, vis_dim)

    features = torch.rand((2, obs_dim))
    frame = torch.rand((2,) + vis_dim)
    action = torch.rand((2, act_dim))

    state = MultiObservation(features, frame)
    value = model(state, action)
    assert list(value.size()) == [2]

    features = torch.rand((obs_dim,))
    frame = torch.rand(vis_dim)
    action = torch.rand((act_dim,))

    state = MultiObservation(features, frame)
    value = model(state, action)
    assert list(value.size()) == [1]
