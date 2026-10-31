"""
Weight data generation script.

This script generates weight data by training models on
MNIST/Fashion-MNIST/CIFAR-10.
Checkpoints are saved at specified iteration intervals.

Usage:
    python -m src.main.generate_training_data [options]
"""

import json

import numpy as np
import torch

from src.cli import parse_generate_args
from src.data import get_cifar10_loaders, get_fashion_mnist_loaders, get_mnist_loaders
from src.generator import WeightGenerator
from src.models import MLP, ResNet18, ResNet20
from src.utils import log_device_info, logger


def generate_single_model(config, seed: int, train_loader, test_loader) -> dict:
    """Generate weight data for a single model.

    Args:
        config: Generation configuration.
        seed: Random seed for this model.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.

    Returns:
        Dictionary containing generation metrics.
    """
    if config.dataset in {"mnist", "fashion_mnist"} and config.arch == "mlp":
        model = MLP(
            hidden_dim=config.model.hidden_dim,
            init_type=config.model.init_type,
            seed=seed,
        )
    elif config.dataset == "cifar10" and config.arch == "resnet20":
        # ResNet-20 uses its own internal initialization.
        model = ResNet20()
    elif config.dataset == "cifar10" and config.arch == "resnet18":
        model = ResNet18()
    else:
        raise ValueError(
            f"Unsupported dataset/arch combination: {config.dataset}/{config.arch}"
        )

    generator = WeightGenerator(model=model, config=config, seed=seed)
    metrics = generator.generate(train_loader, test_loader)

    return metrics


def main() -> None:
    """Main entry point for weight generation."""
    config = parse_generate_args()

    # Log device information
    log_device_info()

    logger.info(f"Loading dataset: {config.dataset} ...")
    if config.dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=config.batch_size)
    elif config.dataset == "fashion_mnist":
        train_loader, test_loader = get_fashion_mnist_loaders(
            batch_size=config.batch_size
        )
    elif config.dataset == "cifar10":
        cifar_root = (
            "data/CIFAR10_resnet18/raw"
            if config.arch == "resnet18"
            else "data/CIFAR10/raw"
        )
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=config.batch_size,
            data_root=cifar_root,
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    logger.info(
        f"Generating weight data for {config.num_seeds} models "
        f"(seeds 0 to {config.num_seeds - 1})"
    )
    logger.info(f"Architecture: {config.arch}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Checkpoint directory: {config.checkpoint_dir}")

    # Collect metrics from all models
    all_metrics = []
    accuracies = []

    for seed in range(config.num_seeds):
        metrics = generate_single_model(config, seed, train_loader, test_loader)
        all_metrics.append(metrics)
        accuracies.append(metrics["best_accuracy"])

    # Calculate statistics
    accuracies_np = np.array(accuracies)
    stats = {
        "n_models": config.num_seeds,
        "accuracies": accuracies,
        "mean_accuracy": float(accuracies_np.mean()),
        "std_accuracy": float(accuracies_np.std()),
        "min_accuracy": float(accuracies_np.min()),
        "max_accuracy": float(accuracies_np.max()),
        "config": {
            "dataset": config.dataset,
            "arch": config.arch,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
            "hidden_dim": config.model.hidden_dim,
            "init_type": config.model.init_type,
        },
    }

    # Log statistics
    logger.info("=" * 60)
    logger.info("Training Data Generation Results:")
    logger.info(f"  Models trained: {config.num_seeds}")
    logger.info(
        f"  Mean Accuracy: {stats['mean_accuracy']:.2f}% ± {stats['std_accuracy']:.2f}%"
    )
    logger.info(f"  Min Accuracy: {stats['min_accuracy']:.2f}%")
    logger.info(f"  Max Accuracy: {stats['max_accuracy']:.2f}%")
    logger.info("=" * 60)

    # Save statistics to JSON
    stats_path = config.checkpoint_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Training statistics saved to {stats_path}")

    logger.info("Weight data generation complete!")


if __name__ == "__main__":
    main()
