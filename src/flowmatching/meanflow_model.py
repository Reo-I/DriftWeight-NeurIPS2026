"""MeanFlow neural network model for weight space generation.

This module provides the neural network architecture for MeanFlow-based
weight generation. The key difference from the standard flow model is that
it conditions on two time variables (t and r) instead of one, enabling
average velocity prediction for efficient few-step generation.

Reference:
    "Mean Flows for One-step Generative Modeling"
    (https://arxiv.org/pdf/2505.13447)
"""

import torch
import torch.nn as nn


class WeightSpaceMeanFlowModel(nn.Module):
    """MeanFlow model for weight space generation with dual time conditioning.

    Predicts the average velocity u(z_t, r, t) for transporting source
    distribution samples to target distribution (trained weights). Unlike
    the standard flow model that predicts instantaneous velocity with a
    single time variable, this model conditions on both t and r.

    The two time variables are embedded separately and combined via
    addition (following the MeanFlow DiT reference implementation).

    Architecture:
        - Separate time embedding MLPs for t and r
        - Main network: Linear -> LayerNorm -> GELU -> Dropout (repeated)
        - Zero-initialized output layer for stability

    Attributes:
        input_dim: Dimension of the weight space.
        hidden_dim: Dimension of hidden layers.
        time_embed_dim: Dimension of each time embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        time_embed_dim: int = 64,
    ) -> None:
        """Initialize WeightSpaceMeanFlowModel.

        Args:
            input_dim: Dimension of the input weight space.
            hidden_dim: Dimension of hidden layers.
            dropout: Dropout probability for regularization.
            time_embed_dim: Dimension of each time embedding.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim

        # Separate time embedding networks for t and r
        self.t_embedder = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.r_embedder = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Main network (same architecture as WeightSpaceFlowModel)
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

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to predict average velocity field u(z_t, r, t).

        Args:
            x: Input tensor of shape (batch_size, input_dim) representing
               points in weight space.
            t: Time tensor of shape (batch_size, 1) with values in [0, 1].
            r: Second time tensor of shape (batch_size, 1) with values
               in [0, 1] where r <= t.

        Returns:
            Predicted average velocity of shape (batch_size, input_dim).
        """
        t_embed = self.t_embedder(t)
        r_embed = self.r_embedder(r)
        # Combine time embeddings via addition (following MeanFlow DiT)
        time_embed = t_embed + r_embed
        combined = torch.cat([x, time_embed], dim=-1)
        return self.net(combined)
