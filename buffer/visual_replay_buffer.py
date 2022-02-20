import torch
import numpy as np

from random import sample
from torch import FloatTensor
from dataclasses import dataclass

from buffer.replay_buffer import ReplayBuffer
from networks.convolutional import MultiObservation


@dataclass(frozen=True)
class VisualBatch:
    states: MultiObservation
    actions: FloatTensor
    rewards: FloatTensor
    next_states: MultiObservation
    done: FloatTensor


class VisualReplayBuffer(ReplayBuffer):
    def __init__(self, size: int, act_dim: int):
        self.state = np.zeros((size,), dtype=MultiObservation)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.next_state = np.zeros((size,), dtype=MultiObservation)
        self.done = np.zeros(size, dtype=np.bool)

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(
        self,
        obs: VisualBatch,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: VisualBatch,
        done: np.ndarray
    ):
        self.state[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.done[self.ptr] = done
        self.next_state[self.ptr] = next_obs
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> VisualBatch:
        idx = sample(range(self.size), batch_size)

        features = torch.stack([x.features for x in self.state[idx]])
        frames = torch.stack([x.frame for x in self.state[idx]])
        state = MultiObservation(features, frames)

        next_features = torch.stack([x.features for x in self.next_state[idx]])
        next_frames = torch.stack([x.frame for x in self.next_state[idx]])
        next_state = MultiObservation(next_features, next_frames)

        return VisualBatch(
            state,
            FloatTensor(self.actions[idx]),
            FloatTensor(self.rewards[idx]),
            next_state,
            FloatTensor(self.done[idx]),
        )
