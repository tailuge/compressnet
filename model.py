"""TinyMLP Architecture Definition.

An 8-layer Multilayer Perceptron with ReLU activations for demonstrating
permutation invariance in neural networks.
"""

import torch
import torch.nn as nn


class TinyMLP(nn.Module):
    """An 8-layer MLP with ReLU activations.

    Architecture:
        - 7 hidden layers with ReLU activations
        - 1 output layer (no activation)
        - Each hidden layer: Linear -> ReLU
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, output_dim: int = 8) -> None:
        """Initialize the TinyMLP architecture.

        Args:
            input_dim: Dimension of input features.
            hidden_dim: Dimension of hidden layers.
            output_dim: Dimension of output predictions.
        """
        super().__init__()

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # 6 intermediate hidden layers
        for _ in range(6):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer (no ReLU)
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # Apply ReLU to all layers except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))

        # Final layer without activation
        x = self.layers[-1](x)
        return x
