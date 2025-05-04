import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads = 8, d_model = 256, masking = True):
        super(MultiHeadAttention, self).__init__() 
        assert d_model % num_heads == 0
        self.masking = masking
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, x):
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
        if self.masking:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf')) 
        attention_scores = torch.softmax(attention_scores, dim = -1)
        attention_output = attention_scores @ v 
        # Combine heads and reshape
        combined = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        combined = self.W_o(combined)
        return combined

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_seq_len=16):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe.unsqueeze(0)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)] 
class TransformerLayer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads = 8, dropout = 0.3):
        super(TransformerLayer, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.attention_module = MultiHeadAttention(num_heads, d_model)
        self.positional_encoding = PositionalEncoding()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential([
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        ])
    def forward(self, embeddings, output_tokens):
        embedded_tokens = self.token_embed(output_tokens)
        inputs = torch.cat([embeddings, embedded_tokens])
        seq_len = inputs.size(1)
        inputs = self.positional_encoding(inputs)
        attention_output = self.attention_module(inputs)
        output = self.layer_norm(attention_output + self.dropout(attention_output)) # Residual skip
        ff_output = self.feed_forward(output)
        return ff_output

