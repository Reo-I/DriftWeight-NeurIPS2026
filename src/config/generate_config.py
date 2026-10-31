"""Configuration for weight data generation using Pydantic."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.utils.device import get_available_device


class ModelConfig(BaseModel):
    """Configuration for MLP model architecture.

    Attributes:
        hidden_dim: Hidden dimension size for MLP layers.
        init_type: Weight initialization type ('xavier', 'he', or 'normal').
    """

    hidden_dim: int = Field(default=32, ge=1, description="Hidden dimension size")
    init_type: Literal["xavier", "he", "normal"] = Field(
        default="he", description="Weight initialization type"
    )


class GenerateConfig(BaseModel):
    """Configuration for weight data generation.

    Attributes:
        num_seeds: Number of models to generate (seeds 0 to num_seeds-1).
        num_epochs: Number of training epochs per model.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        checkpoint_dir: Directory to save model checkpoints.
        save_every_n_iterations: Save checkpoint every N iterations.
        device: Device to use for training (auto-detected by default).
        model: Model architecture configuration.
    """

    # Generation parameters
    dataset: Literal["mnist", "fashion_mnist", "cifar10"] = Field(
        default="mnist", description="Dataset name."
    )
    arch: Literal["mlp", "resnet20", "resnet18"] = Field(
        default="mlp", description="Model architecture."
    )
    num_seeds: int = Field(
        default=100, ge=1, description="Number of models to generate"
    )
    num_epochs: int = Field(default=5, ge=1, description="Number of epochs per model")
    batch_size: int = Field(default=64, ge=1, description="Batch size")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    optimizer: Literal["adam", "adamw", "sgd"] = Field(
        default="adam", description="Optimizer type"
    )

    # Checkpoint parameters
    checkpoint_dir: Path = Field(
        default=Path("data/MNIST/checkpoint"),
        description="Directory to save checkpoints",
    )
    save_every_n_iterations: int = Field(
        default=50, ge=1, description="Save checkpoint every N iterations"
    )

    # Device (auto-detected by default)
    device: str = Field(
        default_factory=get_available_device,
        description="Device for computation (cuda, mps, or cpu)",
    )

    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)

    class Config:
        """Pydantic configuration."""

        frozen = False
