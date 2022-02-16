import torch.nn as nn

from typing import List


def mlp(neurons: List[int]) -> nn.ModuleList:
    layers = nn.ModuleList([
        nn.Linear(x, y) for x, y in zip(neurons[:-1], neurons[1:])
    ])
    return layers
