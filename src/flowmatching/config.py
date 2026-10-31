"""Configuration models for Flow Matching training."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.utils.device import get_available_device


class FlowModelConfig(BaseModel):
    """Configuration for the Flow Matching neural network.

    Attributes:
        hidden_dim: Dimension of the hidden layers in the flow model.
        time_embed_dim: Dimension of the time embedding.
        dropout: Dropout rate for regularization.
    """

    hidden_dim: int = Field(512, description="Hidden dimension for the flow model.")
    time_embed_dim: int = Field(64, description="Dimension of time embedding.")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate.")


class MLPLayerConfig(BaseModel):
    """Configuration for the MLP architecture whose weights are being generated.

    Attributes:
        layer_layout: List of layer sizes [input, hidden1, hidden2, ..., output].
    """

    layer_layout: list[int] = Field(
        [784, 32, 32, 10],
        description="Layer layout for MLP: [input_dim, hidden_dims..., output_dim].",
    )

    @property
    def weight_dim(self) -> int:
        """Calculate total number of parameters (weights + biases)."""
        total = 0
        for i in range(len(self.layer_layout) - 1):
            # Weight matrix: layer_layout[i+1] x layer_layout[i]
            total += self.layer_layout[i] * self.layer_layout[i + 1]
            # Bias vector: layer_layout[i+1]
            total += self.layer_layout[i + 1]
        return total


class FlowMatchingConfig(BaseModel):
    """Configuration for the Flow Matching training process.

    Attributes:
        batch_size: Batch size for training.
        n_iters: Number of training iterations.
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay for regularization.
        sigma: Noise sigma for flow matching.
        patience: Early stopping patience.
        source_std: Standard deviation for source distribution (noise).
        n_samples: Number of samples to generate for evaluation.
        integration_steps: Number of steps for ODE integration during generation.
        integration_method: ODE integration method ('euler' or 'rk4').
        t_dist: Time distribution ('uniform' or 'beta').
        use_pca: Whether to use PCA for dimensionality reduction.
        pca_components: Number of PCA components (only used if use_pca is True).
        checkpoint_dir: Directory to load MLP weight checkpoints from.
        flow_model_save_dir: Directory to save trained flow model.
        device: Device to use for computation.
        log_freq: Frequency of logging during training.
    """

    batch_size: int = Field(8, gt=0, description="Batch size for training.")
    n_iters: int = Field(30000, gt=0, description="Number of training iterations.")
    lr: float = Field(5e-4, gt=0, description="Learning rate.")
    weight_decay: float = Field(1e-5, ge=0, description="Weight decay.")
    sigma: float = Field(0.001, gt=0, description="Noise sigma for flow matching.")
    patience: int = Field(100, gt=0, description="Early stopping patience.")
    source_std: float = Field(
        0.01, gt=0, description="Standard deviation of source noise."
    )
    n_samples: int = Field(100, gt=0, description="Number of samples to generate.")
    integration_steps: int = Field(100, gt=0, description="Steps for ODE integration.")
    integration_method: Literal["euler", "rk4"] = Field(
        "rk4", description="ODE integration method."
    )
    t_dist: Literal["uniform", "beta"] = Field(
        "uniform", description="Time distribution for sampling."
    )

    # Dimensionality reduction options
    use_pca: bool = Field(False, description="Whether to use PCA.")
    pca_components: int | None = Field(
        None, description="Number of PCA components (None = auto)."
    )

    # Directories
    checkpoint_dir: Path = Field(
        Path("data/MNIST/checkpoint"),
        description="Directory containing trained MLP checkpoints.",
    )
    flow_model_save_dir: Path = Field(
        Path("data/MNIST/flow_model"),
        description="Directory to save trained flow model.",
    )

    # Hardware (auto-detected by default)
    device: str = Field(
        default_factory=get_available_device,
        description="Device for computation (cuda, mps, or cpu).",
    )
    log_freq: int = Field(100, gt=0, description="Logging frequency.")

    # Model configurations (nested)
    flow_model: FlowModelConfig = Field(
        default_factory=FlowModelConfig, description="Flow model configuration."
    )
    mlp_layer: MLPLayerConfig = Field(
        default_factory=MLPLayerConfig, description="MLP layer configuration."
    )


class GenerationConfig(BaseModel):
    """Configuration for generating new model weights using a trained flow model.

    Attributes:
        flow_model_path: Path to the trained flow model checkpoint.
        n_samples: Number of new weights to generate.
        integration_steps: Number of ODE integration steps.
        integration_method: ODE integration method.
        source_std: Standard deviation of source noise.
        device: Device for computation.
        output_dir: Directory to save generated weights.
    """

    flow_model_path: Path = Field(
        ..., description="Path to trained flow model checkpoint."
    )
    n_samples: int = Field(100, gt=0, description="Number of weights to generate.")
    integration_steps: int = Field(100, gt=0, description="ODE integration steps.")
    integration_method: Literal["euler", "rk4"] = Field(
        "rk4", description="Integration method."
    )
    source_std: float = Field(0.01, gt=0, description="Source noise std.")
    device: str = Field(
        default_factory=get_available_device,
        description="Device for computation (cuda, mps, or cpu).",
    )
    output_dir: Path = Field(
        Path("data/MNIST/generated_weights"),
        description="Output directory for generated weights.",
    )
    mlp_layer: MLPLayerConfig = Field(
        default_factory=MLPLayerConfig, description="MLP layer configuration."
    )
    flow_model: FlowModelConfig = Field(
        default_factory=FlowModelConfig, description="Flow model configuration."
    )
