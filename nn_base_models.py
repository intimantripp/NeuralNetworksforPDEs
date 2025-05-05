import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_units, activation='tanh'):
        super(FeedforwardNN, self).__init__()
        
        # Activation function lookup
        activations = {
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'sine': lambda: torch.sin,  # Custom sine activation
            'gelu': nn.GELU,
            'swish': nn.SiLU  # Swish is also called SiLU in PyTorch
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation function: {activation}")
        self.activation_fn = activations[activation]()

        # Construct layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(self.activation_fn)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(self.activation_fn)
        layers.append(nn.Linear(hidden_units, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)