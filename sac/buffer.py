import numpy as np

from random import sample
from dataclasses import dataclass

@dataclass(frozen=True)
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    done: np.ndarray


class ReplayBuffer(object):
    def __init__(self, size, obs_dim, act_dim):
        self.state = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_state = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros(size, dtype=np.bool)

        self.ptr = 0
        self.size = size

    def store(self, obs, act, rew, next_obs, done):
        assert self.ptr < self.size
        self.state[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.next_state[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr += 1

    def sample(self, batch_size):
        assert self.ptr >= batch_size, "Number of samples less than batch size."
        assert self.ptr <= self.size, "Number of samples must be at most buffer size."

        idx = sample(range(self.ptr), batch_size)

        return Batch(
            self.state[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_state[idx],
            self.done[idx]
        )

