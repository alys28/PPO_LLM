from dataclasses import dataclass
import torch

@dataclass
class Trajectory:
    states: torch.Tensor
    actions: torch.Tensor
    advantage_estimates: torch.Tensor
    rewards: torch.Tensor


class RolloutBuffer:

    def __init__(self, capacity):
        self.cap = capacity