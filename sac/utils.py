import torch
import mlflow
import numpy as np

from typing import Dict, Any
from torch import FloatTensor
from abc import ABC, abstractmethod


class StateNormalizer(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def normalize_state(self, state: np.ndarray, update: bool = True, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_state(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load_state(self, auxiliaries: Dict[str, Any]):
        raise NotImplementedError


class WelfordVarianceEstimate(StateNormalizer):
    def __init__(self, mean: FloatTensor = None, variance: FloatTensor = None, step: int = 1):
        self.mean = mean
        self.variance = variance
        self.step = step

    def normalize_state(self, state: np.ndarray, update: bool = True):
        """
        Welford's algorithm to normalize a state, and optionally update the
        statistics for normalizing states using the new state online.
        """
        state = FloatTensor(state)

        if self.step == 1:
            self.mean = torch.zeros(state.size(-1))
            self.variance = torch.ones(state.size(-1))

        if update:
            state_old = self.mean
            self.mean += (state - state_old) / self.step
            self.variance += (state - state_old) * (state - state_old)
            self.step += 1

        numerator = state - self.mean
        denominator = torch.sqrt(self.variance / self.step)
        normalized_state = numerator / denominator
        return normalized_state

    def save_state(self, path: str):
        mlflow.pytorch.log_state_dict({
            "welford_mean": self.mean,
            "welford_step": self.step,
            "welford_variance": self.variance,
        }, artifact_path=path)

    def load_state(self, state_dict: Dict[str, Any]):
        self.mean = state_dict["welford_mean"]
        self.variance = state_dict["welford_variance"]
        self.step = state_dict["welford_step"]


class Identity(StateNormalizer):
    def __init__(self):
        super().__init__()

    def normalize_state(self, state: np.ndarray, update: bool = True):
        return torch.FloatTensor(state)

    def save_state(self, path: str):
        pass

    def load_state(self, state_dict: Dict[str, Any]):
        pass
