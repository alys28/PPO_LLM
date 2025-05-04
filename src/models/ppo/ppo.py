import torch
import numpy as np
from torch import nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformerLayer import TransformerLayer
from reward import RewardBuilder
class PPO(nn.Module):
    def __init__(self, total_timesteps):
        self.total_timesteps = total_timesteps
    def computeGAE(self):
        pass
    def step(self):
        pass
    def collectRollout(self):
        pass


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, output_dim = 1):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential([
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        ])
    def forward(self, x):
        return self.network(x)
    
class PolicyNetwork(nn.Module):
    def __init__(self, hidden_dim, output_dim = 11):
        self.transformerLayer = TransformerLayer(hidden_dim)
        self.network = nn.Sequential([
            self.transformerLayer,
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ])
   
    def forward(self, x):
        return self.network(x)