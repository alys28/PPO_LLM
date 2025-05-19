import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads = 8, d_model = 256):
        super(MultiHeadAttention, self).__init__() 
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, x, causal_mask = None, key_padding_mask = None):
        batch_size, seq_len, d_in = x.size()
        assert d_in == self.d_model, "Embedding Dimension is Incompatible with Initilized"
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # Splitting heads
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        attention_scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k) # seq_len * seq_len
        if causal_mask is not None:
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float('-inf'))
        if key_padding_mask is not None:
            # print("attention_scores", attention_scores.shape)
            # print("key_padding_mask", key_padding_mask.shape)
            attention_scores = attention_scores.masked_fill(key_padding_mask == 0, float('-inf'))
        attention_scores = torch.softmax(attention_scores, dim = -1)
        attention_output = attention_scores @ v 
        # Combine heads and reshape
        combined = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        combined = self.W_o(combined)
        return combined

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_seq_len=16):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
class TransformerLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, num_heads = 8, dropout = 0.3, num_transformer_layers = 2):
        super(TransformerLayer, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout) 
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(num_heads, d_model),
                'norm1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Linear(d_model * 2, d_model)
                ),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_transformer_layers)
        ])
        
        self.num_transformer_layers = num_transformer_layers

    def forward(self, embeddings, output_tokens, causal_mask=None, key_padding_mask=None):
        # Only embed tokens if we receive integer indices
        if output_tokens.dtype in [torch.long, torch.int]:
            embedded_tokens = self.token_embed(output_tokens)
            inputs = torch.cat([embeddings.unsqueeze(1), embedded_tokens], dim=1)
        else:
            # If we receive float tensors (from previous layer), use them directly
            inputs = torch.cat([embeddings.unsqueeze(1), output_tokens], dim=1)
        inputs = self.positional_encoding(inputs)
        
        for layer in self.layers:
            attention_output = layer['attention'](inputs, causal_mask, key_padding_mask)
            inputs = layer['norm1'](inputs + self.dropout(attention_output))
            
            ff_output = layer['ff'](inputs)
            inputs = layer['norm2'](inputs + self.dropout(ff_output))
            
        return inputs[:, embeddings.unsqueeze(1).size(1):, :]  # Only return predictions for output tokens


