import torch
from torch import nn
import sys
import os
from torch.nn.functional import softmax
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformerLayer import TransformerLayer


class SFT_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_dim, max_seq_len, num_heads = 8, num_transformer_layers = 2):
        super(SFT_Model, self).__init__()
        self.num_transformer_layers = num_transformer_layers
        self.transformer_layers = nn.ModuleList([TransformerLayer(vocab_size, input_dim, max_seq_len, num_heads) for _ in range(num_transformer_layers)])
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, vocab_size)
        )
        self.layer_for_embeddings = nn.Sequential(
            nn.Linear(embedding_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
        )
    
    def forward(self, input_embedding, output_tokens, causal_mask=None, key_padding_mask=None):
        x = output_tokens
        input_embedding = self.layer_for_embeddings(input_embedding)
        for layer in self.transformer_layers:
            x = layer(input_embedding, x, causal_mask, key_padding_mask)
        logits = self.layers(x)
        return logits