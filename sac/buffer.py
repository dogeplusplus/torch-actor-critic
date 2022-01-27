import numpy as np

from random import sample
from dataclasses import dataclass
from torch import FloatTensor
from torch.cuda import FloatTensor as FloatCudaTensor

from typing import Union


@dataclass(frozen=True)
class Batch:
    states: Union[FloatTensor, FloatCudaTensor]
    actions: Union[FloatTensor, FloatCudaTensor]
    rewards: Union[FloatTensor, FloatCudaTensor]
    next_states: Union[FloatTensor, FloatCudaTensor]
    done: Union[FloatTensor, FloatCudaTensor]


class ReplayBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, device: str):
        self.state = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_state = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros(size, dtype=np.bool)

        self.ptr = 0
        self.size = size
        self.device = device

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray
    ):
        assert self.ptr < self.size
        self.state[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.next_state[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr += 1

    def sample(self, batch_size) -> Batch:
        assert self.ptr >= batch_size, "Number of samples less than batch size."
        assert self.ptr <= self.size, "Number of samples must be at most buffer size."

        idx = sample(range(self.ptr), batch_size)

        return Batch(
            FloatTensor(self.state[idx]).to(self.device),
            FloatTensor(self.actions[idx]).to(self.device),
            FloatTensor(self.rewards[idx]).to(self.device),
            FloatTensor(self.next_state[idx]).to(self.device),
            FloatTensor(self.done[idx]).to(self.device),
        )

