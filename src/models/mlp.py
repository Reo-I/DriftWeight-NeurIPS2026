"""Multi-Layer Perceptron model for MNIST classification."""

from typing import Literal

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron for MNIST digit classification.

    A simple feedforward neural network with two hidden layers.

    Attributes:
        fc1: First fully connected layer (784 -> hidden_dim).
        fc2: Second fully connected layer (hidden_dim -> hidden_dim).
        fc3: Output layer (hidden_dim -> 10).
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        init_type: Literal["xavier", "he", "normal"] = "he",
        seed: int | None = None,
    ) -> None:
        """Initialize the MLP model.

        Args:
            hidden_dim: Size of hidden layers.
            init_type: Weight initialization type ('xavier', 'he', or 'normal').
            seed: Random seed for weight initialization. If None, no seed is set.
        """
        super().__init__()

        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)

        if seed is not None:
            torch.manual_seed(seed)

        self._init_weights(init_type)

    def _init_weights(self, init_type: Literal["xavier", "he", "normal"]) -> None:
        """Initialize model weights.

        Args:
            init_type: Initialization type to use.
        """
        for layer in [self.fc1, self.fc2, self.fc3]:
            if init_type == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif init_type == "he":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            else:  # normal
                nn.init.normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784).

        Returns:
            Output logits of shape (batch_size, 10).
        """
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
