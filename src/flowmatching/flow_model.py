"""Flow Matching neural network models for weight space generation.

This module provides neural network architectures that learn the vector field
for flow matching in weight space.
"""

import torch
import torch.nn as nn


class WeightSpaceFlowModel(nn.Module):
    """Standard flow model for weight space generation.

    A neural network that learns to predict the velocity field v_t(x)
    for flow matching. Given a point x_t in weight space and time t,
    it predicts the velocity to move towards the target distribution.

    Architecture:
        - Time embedding MLP
        - Main network: Linear -> LayerNorm -> GELU -> Dropout (repeated)
        - Zero-initialized output layer for stability

    Attributes:
        input_dim: Dimension of the weight space.
        hidden_dim: Dimension of hidden layers.
        time_embed_dim: Dimension of time embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        time_embed_dim: int = 64,
    ) -> None:
        """Initialize WeightSpaceFlowModel.

        Args:
            input_dim: Dimension of the input weight space.
            hidden_dim: Dimension of hidden layers.
            dropout: Dropout probability for regularization.
            time_embed_dim: Dimension of time embedding.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Main network
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Zero initialization for output layer (stability)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict velocity field.

        Args:
            x: Input tensor of shape (batch_size, input_dim) representing
               points in weight space.
            t: Time tensor of shape (batch_size, 1) with values in [0, 1].

        Returns:
            Predicted velocity field of shape (batch_size, input_dim).
        """
        t_embed = self.time_embed(t)
        combined = torch.cat([x, t_embed], dim=-1)
        return self.net(combined)


class Bunch:
    """Simple attribute container for flow matching data.

    Provides a convenient way to pass multiple tensors together.

    Attributes:
        Any keyword arguments passed to __init__ become attributes.

    Example:
        >>> flow = Bunch(t=t_tensor, x0=x0, xt=xt, x1=x1, ut=ut)
        >>> print(flow.t.shape)
    """

    def __init__(self, **kwargs) -> None:
        """Initialize with keyword arguments as attributes."""
        self.__dict__.update(kwargs)
