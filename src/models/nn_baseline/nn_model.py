import torch
from torch import nn


class NNModel(nn.Module):

    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        assert len(hidden_dims) > 0, "Need an output dimension"
        assert hidden_dims[-1] == 1, "NN built for regression task"
        layer_dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
                if i % 3 == 0:
                    layers.append(nn.Dropout(0.3))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

