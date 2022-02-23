import torch


from networks.linear import Actor, Critic, DoubleCritic


def test_linear_actor():
    obs_dim = 10
    act_dim = 20
    hidden_sizes = [256, 256]
    actor = Actor(obs_dim, act_dim, hidden_sizes)
    state = torch.ones((10,))

    pi, log_prob = actor(state)
    assert list(pi.size()) == [20]
    assert list(log_prob.size()) == []


def test_linear_critic():
    obs_dim = 5
    act_dim = 10
    hidden_sizes = [256, 256]
    critic = Critic(obs_dim, act_dim, hidden_sizes)

    state = torch.ones((2, obs_dim))
    action = torch.ones((2, act_dim))
    value = critic(state, action)
    assert list(value.size()) == [2]


def test_double_critic():
    obs_dim = 5
    act_dim = 10
    hidden_sizes = [256, 256]
    critic = DoubleCritic(obs_dim, act_dim, hidden_sizes)

    state = torch.ones((3, obs_dim))
    action = torch.ones((3, act_dim))
    value_1, value_2 = critic(state, action)
    assert list(value_1.size()) == [3]
    assert list(value_2.size()) == [3]
