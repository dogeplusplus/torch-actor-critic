import torch

from networks.convolutional import MultiModalActor
from environments.wall_runner import MultiObservation


def test_multimodal():
    obs_dim = 20
    act_dim = 10
    vis_dim = (3, 64, 64)
    model = MultiModalActor(obs_dim, act_dim, vis_dim)

    features = torch.rand((obs_dim,))
    camera = torch.rand(vis_dim)
    x = MultiObservation(features, camera)
    pi, log_prob = model(x)

    assert list(pi.size()) == [act_dim]
    assert list(log_prob.size()) == []
