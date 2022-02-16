import numpy as np

from random import sample
from torch import FloatTensor
from dataclasses import dataclass


@dataclass(frozen=True)
class Batch:
    states: FloatTensor
    actions: FloatTensor
    rewards: FloatTensor
    next_states: FloatTensor
    done: FloatTensor


class ReplayBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int):
        self.state = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_state = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros(size, dtype=np.bool)

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray
    ):
        self.state[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.next_state[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        idx = sample(range(self.size), batch_size)

        return Batch(
            FloatTensor(self.state[idx]),
            FloatTensor(self.actions[idx]),
            FloatTensor(self.rewards[idx]),
            FloatTensor(self.next_state[idx]),
            FloatTensor(self.done[idx]),
        )
