"""Argument parser for weight data generation script."""

import argparse
from pathlib import Path

from src.config import GenerateConfig, ModelConfig
from src.utils.device import get_available_device


def parse_generate_args() -> GenerateConfig:
    """Parse command line arguments for weight generation.

    Returns:
        GenerateConfig object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate weight data by training models on MNIST/Fashion-MNIST/CIFAR-10"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Compute device (default: auto selects CUDA, MPS, or CPU).",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion_mnist", "cifar10"],
        default="mnist",
        help="Dataset to use (default: mnist)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["mlp", "resnet20", "resnet18"],
        default="mlp",
        help="Model architecture (default: mlp)",
    )

    # Generation parameters
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=100,
        help="Number of models to generate (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs per model (default: dataset-dependent)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Batch size (default: dataset-dependent: 64 for mnist/mlp, "
            "128 for fashion_mnist/mlp and cifar10/resnet20)"
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: dataset/arch-dependent)",
    )

    # Model parameters
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension size for MLP (default: dataset-dependent)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd"],
        default=None,
        help="Optimizer type (default: dataset-dependent)",
    )
    parser.add_argument(
        "--init-type",
        type=str,
        choices=["xavier", "he", "normal"],
        default="he",
        help="Weight initialization type (default: he)",
    )

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory to save checkpoints "
            "(default: data/MNIST/checkpoint, data/FashionMNIST/checkpoint, "
            "data/CIFAR10/checkpoint, or data/CIFAR10_resnet18/checkpoint)"
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N iterations (default: dataset/arch-dependent)",
    )

    args = parser.parse_args()

    if args.dataset in {"mnist", "fashion_mnist"} and args.arch != "mlp":
        raise ValueError(
            "MNIST/Fashion-MNIST currently supports --arch mlp only."
        )
    if args.dataset == "cifar10" and args.arch not in {"resnet20", "resnet18"}:
        raise ValueError("CIFAR-10 currently supports --arch resnet20/resnet18 only.")

    if args.dataset == "fashion_mnist":
        default_epochs = 25
        default_batch_size = 128
        default_hidden_dim = 128
        default_optimizer = "adamw"
        default_save_every = 100
    elif args.dataset == "cifar10":
        if args.arch == "resnet18":
            # DeepWeightFlow ResNet-18 data-generation recipe
            default_epochs = 100
            default_batch_size = 128
            default_hidden_dim = 32
            default_optimizer = "sgd"
            default_lr = 0.1
            default_save_every = 1000
        else:
            default_epochs = 5
            default_batch_size = 128
            default_hidden_dim = 32
            default_optimizer = "adam"
            default_lr = 0.001
            default_save_every = 100
    elif args.dataset == "mnist":
        default_epochs = 5
        default_batch_size = 64
        default_hidden_dim = 32
        default_optimizer = "adam"
        default_lr = 0.001
        default_save_every = 100
    else:
        raise ValueError(
            f"Unsupported dataset for defaults: {args.dataset}"
        )
    if args.dataset == "fashion_mnist":
        default_lr = 0.001

    num_epochs = args.epochs if args.epochs is not None else default_epochs
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    hidden_dim = (
        args.hidden_dim if args.hidden_dim is not None else default_hidden_dim
    )
    optimizer = args.optimizer if args.optimizer is not None else default_optimizer
    learning_rate = args.lr if args.lr is not None else default_lr
    save_every = args.save_every if args.save_every is not None else default_save_every

    if args.checkpoint_dir is None:
        if args.dataset == "cifar10" and args.arch == "resnet18":
            checkpoint_dir = Path("data/CIFAR10_resnet18/checkpoint")
        elif args.dataset == "cifar10":
            checkpoint_dir = Path("data/CIFAR10/checkpoint")
        elif args.dataset == "fashion_mnist":
            checkpoint_dir = Path("data/FashionMNIST/checkpoint")
        else:
            checkpoint_dir = Path("data/MNIST/checkpoint")
    else:
        checkpoint_dir = Path(args.checkpoint_dir)

    return GenerateConfig(
        dataset=args.dataset,
        arch=args.arch,
        num_seeds=args.num_seeds,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        save_every_n_iterations=save_every,
        device=get_available_device() if args.device == "auto" else args.device,
        model=ModelConfig(
            hidden_dim=hidden_dim,
            init_type=args.init_type,
        ),
    )
