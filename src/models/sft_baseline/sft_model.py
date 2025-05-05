import torch
from torch import nn
import sys
import os
from torch.nn.functional import softmax
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformerLayer import TransformerLayer


class SFT_Model(nn.Module):
    def __init__(self, vocab_size, input_dim, num_heads = 8):
        super(SFT_Model, self).__init__()
        self.layers = nn.Sequential(
            TransformerLayer(vocab_size, input_dim, num_heads),  # Must return (B, T, D)
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, vocab_size)
        )
    
    def forward(self, x, return_logits = True):
        return self.layers(x) if return_logits else softmax(self.layers(x), dim = -1)