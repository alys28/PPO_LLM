import torch
from torch import nn
import sys
import os
from torch.nn.functional import softmax
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformerLayer import TransformerLayer


class SFT_Model(nn.Module):
    def __init__(self, vocab_size, input_dim, max_seq_len, num_heads = 8):
        super(SFT_Model, self).__init__()
        self.transformer = TransformerLayer(vocab_size, input_dim, max_seq_len, num_heads)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, vocab_size)
        )
    
    def forward(self, input_embedding, output_tokens, causal_mask=None, key_padding_mask=None):
        x = self.transformer(input_embedding, output_tokens, causal_mask, key_padding_mask)
        logits = self.layers(x)
        return logits