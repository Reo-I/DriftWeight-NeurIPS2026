"""Argument parser for Flow Matching / MeanFlow training script.

Supports two generative methods:
    - flowmatching: Standard Conditional Flow Matching (DeepWeightFlow)
    - meanflow: MeanFlow with average velocity (few-step generation)
"""

import argparse
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.utils.device import get_available_device


class FlowTrainingArgs(BaseModel):
    """Configuration for Flow Matching / MeanFlow training.

    Attributes:
        method: Generative method to use ('flowmatching' or 'meanflow').
        n_iters: Number of training iterations.
        batch_size: Batch size for training.
        lr: Learning rate.
        weight_decay: Weight decay for regularization.
        hidden_dim: Hidden dimension for flow model.
        time_embed_dim: Time embedding dimension.
        dropout: Dropout rate.
        sigma: Noise sigma for flow matching.
        source_std: Source noise standard deviation.
        t_dist: Time distribution (for flowmatching).
        n_samples: Number of models to generate for evaluation.
        integration_method: ODE integration method (for flowmatching).
        checkpoint_dir: Directory containing trained MLP checkpoints.
        save_dir: Directory to save trained flow model.
        device: Device to use (auto-detected by default).
        use_pca: Whether to use PCA.
        pca_components: Number of PCA components (None = auto).
        log_freq: Logging frequency.
        save_generated_models: Whether to save generated model weights.
        mf_flow_ratio: MeanFlow ratio of r=t samples (FM learning signal).
        mf_t1_ratio: MeanFlow ratio of samples forced to t=1.
        mf_r0_ratio: MeanFlow ratio of samples forced to r=0.
        mf_onestep_lambda: Weight for 1-step auxiliary loss.
        mf_time_dist: MeanFlow time distribution type.
        mf_time_dist_mu: MeanFlow lognorm distribution mean.
        mf_time_dist_sigma: MeanFlow lognorm distribution std.
        mf_loss_gamma: MeanFlow adaptive loss gamma parameter.
        mf_jvp_api: MeanFlow JVP computation backend.
        eval_steps: List of step sizes for generation/evaluation.
        use_drift_loss: Whether to enable drifting loss.
    """

    # Method selection
    dataset: Literal["mnist", "fashion_mnist", "cifar10"] = Field(
        "mnist", description="Dataset to train/evaluate on."
    )
    arch: Literal["mlp", "resnet20", "resnet18"] = Field(
        "mlp", description="Architecture whose weights are modeled."
    )

    method: Literal["flowmatching", "meanflow"] = Field(
        "flowmatching",
        description="Generative method: 'flowmatching' (DeepWeightFlow) "
        "or 'meanflow' (MeanFlow, few-step generation).",
    )

    # Training parameters
    n_iters: int = Field(30000, gt=0, description="Number of training iterations.")
    batch_size: int = Field(8, gt=0, description="Batch size for training.")
    lr: float = Field(5e-4, gt=0, description="Learning rate.")
    weight_decay: float = Field(1e-5, ge=0, description="Weight decay.")

    # Flow model parameters
    hidden_dim: int = Field(512, gt=0, description="Hidden dimension for flow model.")
    time_embed_dim: int = Field(64, gt=0, description="Time embedding dimension.")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate.")

    # Flow matching parameters (shared / flowmatching-specific)
    sigma: float = Field(0.001, gt=0, description="Noise sigma for flow matching.")
    source_std: float = Field(0.01, gt=0, description="Source noise std.")
    t_dist: Literal["uniform", "beta"] = Field(
        "uniform", description="Time distribution (flowmatching only)."
    )

    # Generation parameters
    n_samples: int = Field(100, gt=0, description="Number of samples to generate.")
    integration_method: Literal["euler", "rk4"] = Field(
        "rk4", description="Integration method (flowmatching only)."
    )

    # Paths
    checkpoint_dir: Path = Field(
        Path("data/MNIST/checkpoint"),
        description="Directory containing trained model checkpoints.",
    )
    save_dir: Path = Field(
        Path("results"),
        description="Directory to save outputs (flow model, generated models, results).",
    )

    # Hardware (auto-detected by default)
    device: str = Field(
        default_factory=get_available_device,
        description="Device to use (cuda, mps, or cpu).",
    )

    # PCA options (critical for matching paper results!)
    use_pca: bool = Field(
        True,
        description="Use PCA dimensionality reduction. "
        "IMPORTANT: Paper uses PCA with 99 components for 100 models.",
    )
    pca_components: int | None = Field(
        None,
        description="Number of PCA components. "
        "None = auto (n_fit - 1). Explicitly set to override.",
    )
    pca_fit_source: Literal["final", "all"] = Field(
        "final",
        description="Which checkpoints to fit PCA on. "
        "'final': final.pt only (default). "
        "'all': final + history checkpoints (requires --use-drift-loss).",
    )

    # Other options
    log_freq: int = Field(100, gt=0, description="Logging frequency.")
    save_generated_models: bool = Field(
        True, description="Whether to save generated model weights."
    )

    # ===== MeanFlow-specific parameters =====
    mf_flow_ratio: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="MeanFlow: fraction of batch where r=t "
        "(instantaneous velocity learning). Paper best: 0.75 "
        "(25% r!=t in Table 1a).",
    )
    mf_t1_ratio: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="MeanFlow: fraction of batch forced to t=1.",
    )
    mf_r0_ratio: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="MeanFlow: fraction of batch forced to r=0.",
    )
    mf_onestep_lambda: float = Field(
        1.0,
        ge=0.0,
        description="MeanFlow: weight of 1-step auxiliary loss.",
    )
    mf_time_dist: Literal["uniform", "lognorm"] = Field(
        "uniform",
        description="MeanFlow: time distribution for sampling (t, r).",
    )
    mf_time_dist_mu: float = Field(
        -0.4,
        description="MeanFlow: mean for lognorm time distribution.",
    )
    mf_time_dist_sigma: float = Field(
        1.0,
        description="MeanFlow: std for lognorm time distribution.",
    )
    mf_loss_gamma: float = Field(
        0.0,
        ge=0.0,
        description="MeanFlow: gamma for adaptive L2 loss. "
        "p=1-gamma. Paper best: gamma=0.0 (p=1.0, Table 1e). "
        "gamma=0.5 is Pseudo-Huber (p=0.5).",
    )
    mf_jvp_api: Literal["autograd", "functorch"] = Field(
        "autograd",
        description="MeanFlow: JVP computation backend.",
    )

    # ===== Evaluation parameters =====
    eval_steps: list[int] = Field(
        [5],
        description="List of step sizes for generation/evaluation. "
        "Applies to both flowmatching and meanflow. "
        "E.g., [1, 2, 3, 5, 20, 100] to compare multiple step sizes.",
    )

    # ===== Drifting loss parameters =====
    use_drift_loss: bool = Field(
        False, description="Enable Drifting Models style drift loss."
    )
    drift_lambda: float = Field(
        1.0, ge=0.0, description="Base multiplier for drifting loss."
    )
    drift_target_ratio: float = Field(
        0.1,
        ge=0.0,
        description="Target drift contribution ratio relative to base loss.",
    )
    drift_auto_scale: Literal["ema", "grad", "none"] = Field(
        "ema", description="Automatic drift scaling strategy."
    )
    drift_temperatures: list[float] = Field(
        [0.02, 0.05, 0.2],
        description="Temperature list for compute_V aggregation.",
    )
    drift_neg_max_total: int = Field(
        32, gt=0, description="Max total negatives for compute_V."
    )
    drift_same_ratio: float = Field(
        0.2, ge=0.0, le=1.0, description="Ratio of same-seed negatives in history."
    )
    drift_history_tail_exclude: float = Field(
        0.05,
        ge=0.0,
        lt=1.0,
        description="Exclude latest tail ratio of trajectory history from sampling.",
    )
    drift_use_a6_norm: bool = Field(
        True,
        description="Enable A.6 feature/drift normalization for drifting loss.",
    )
    drift_a6_eps: float = Field(
        1e-6, gt=0.0, description="Numerical epsilon used by A.6 normalization."
    )
    drift_hardneg_k: int = Field(
        64, ge=0, description="Hard-negative top-k count. 0=disabled (use all negatives)."
    )
    drift_ema_decay: float = Field(
        0.99, gt=0.0, lt=1.0, description="EMA decay for loss-scale auto tuning."
    )
    drift_scale_eps: float = Field(
        1e-8, gt=0.0, description="Numerical epsilon for drift scaling."
    )


def parse_flow_args() -> FlowTrainingArgs:
    """Parse command-line arguments for Flow Matching / MeanFlow training.

    Returns:
        FlowTrainingArgs object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train Flow Matching or MeanFlow model for weight generation"
    )

    # ===== Method selection =====
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Dataset to use (default: mnist)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="mlp",
        choices=["mlp", "resnet20", "resnet18"],
        help="Model architecture (default: mlp)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="flowmatching",
        choices=["flowmatching", "meanflow"],
        help="Generative method: 'flowmatching' or 'meanflow' (default: flowmatching)",
    )

    # ===== Training parameters =====
    parser.add_argument(
        "--n-iters",
        type=int,
        default=30000,
        help="Number of training iterations (default: 30000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5)",
    )

    # ===== Flow model parameters =====
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for flow model (default: 512)",
    )
    parser.add_argument(
        "--time-embed-dim",
        type=int,
        default=64,
        help="Time embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )

    # ===== Flow matching parameters =====
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.001,
        help="Noise sigma for flow matching (default: 0.001)",
    )
    parser.add_argument(
        "--source-std",
        type=float,
        default=0.01,
        help="Source noise standard deviation (default: 0.01)",
    )
    parser.add_argument(
        "--t-dist",
        type=str,
        default="uniform",
        choices=["uniform", "beta"],
        help="Time distribution for flowmatching (default: uniform)",
    )

    # ===== Generation parameters =====
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of models to generate for evaluation (default: 100)",
    )
    parser.add_argument(
        "--integration-method",
        type=str,
        default="rk4",
        choices=["euler", "rk4"],
        help="ODE integration method for flowmatching (default: rk4)",
    )

    # ===== Paths =====
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Directory containing trained model checkpoints "
            "(default follows dataset)"
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory to save outputs (uses $RESULTS_DIR if set)",
    )

    # ===== PCA options =====
    parser.add_argument(
        "--use-pca",
        action="store_true",
        default=True,
        help="Use PCA dimensionality reduction (default: True, paper uses this)",
    )
    parser.add_argument(
        "--no-pca",
        action="store_false",
        dest="use_pca",
        help="Disable PCA (not recommended, significantly worse results)",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help=(
            "Number of PCA components. "
            "Default: None (auto = n_fit - 1, e.g. 99 for 100 models). "
            "Explicitly set to override (e.g. 99)."
        ),
    )
    parser.add_argument(
        "--pca-fit-source",
        type=str,
        choices=["final", "all"],
        default="final",
        help=(
            "Which checkpoints to fit PCA on. "
            "'final': final.pt only (default). "
            "'all': final + history checkpoints (requires --use-drift-loss)."
        ),
    )

    # ===== Other options =====
    parser.add_argument(
        "--log-freq",
        type=int,
        default=100,
        help="Logging frequency (default: 100)",
    )
    parser.add_argument(
        "--save-generated-models",
        action="store_true",
        default=True,
        help="Save generated model weights (default: True)",
    )
    parser.add_argument(
        "--no-save-generated-models",
        action="store_false",
        dest="save_generated_models",
        help="Do not save generated model weights",
    )

    # ===== MeanFlow-specific parameters =====
    parser.add_argument(
        "--mf-flow-ratio",
        type=float,
        default=0.75,
        help="MeanFlow: fraction of batch where r=t (default: 0.75, paper best)",
    )
    parser.add_argument(
        "--mf-t1-ratio",
        type=float,
        default=0.0,
        help="MeanFlow: fraction of batch forced to t=1 (default: 0.0)",
    )
    parser.add_argument(
        "--mf-r0-ratio",
        type=float,
        default=0.0,
        help="MeanFlow: fraction of batch forced to r=0 (default: 0.0)",
    )
    parser.add_argument(
        "--mf-onestep-lambda",
        type=float,
        default=1.0,
        help="MeanFlow: 1-step auxiliary loss weight (default: 1.0)",
    )
    parser.add_argument(
        "--mf-time-dist",
        type=str,
        default="uniform",
        choices=["uniform", "lognorm"],
        help="MeanFlow: time distribution (default: uniform)",
    )
    parser.add_argument(
        "--mf-time-dist-mu",
        type=float,
        default=-0.4,
        help="MeanFlow: lognorm distribution mean (default: -0.4)",
    )
    parser.add_argument(
        "--mf-time-dist-sigma",
        type=float,
        default=1.0,
        help="MeanFlow: lognorm distribution std (default: 1.0)",
    )
    parser.add_argument(
        "--mf-loss-gamma",
        type=float,
        default=0.0,
        help="MeanFlow: adaptive loss gamma, p=1-gamma (default: 0.0, p=1.0 paper best)",
    )
    parser.add_argument(
        "--mf-jvp-api",
        type=str,
        default="autograd",
        choices=["autograd", "functorch"],
        help="MeanFlow: JVP computation backend (default: autograd)",
    )

    # ===== Evaluation parameters =====
    parser.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        default=[5],
        help="Step sizes for generation/evaluation "
        "(e.g., --eval-steps 1 2 3 5 20 100). "
        "Multiple values will evaluate each step size separately (default: 5).",
    )
    parser.add_argument(
        "--use-drift-loss",
        action="store_true",
        default=False,
        help="Enable drifting loss branch (default: False)",
    )
    parser.add_argument(
        "--drift-lambda",
        type=float,
        default=1.0,
        help="Base multiplier for drifting loss (default: 1.0)",
    )
    parser.add_argument(
        "--drift-target-ratio",
        type=float,
        default=0.1,
        help="Target drift-to-base contribution ratio (default: 0.1)",
    )
    parser.add_argument(
        "--drift-auto-scale",
        type=str,
        default="ema",
        choices=["ema", "grad", "none"],
        help="Drift auto scaling method: ema/grad/none (default: ema)",
    )
    parser.add_argument(
        "--drift-temperatures",
        type=str,
        default="0.02,0.05,0.2",
        help="Comma-separated temperature list for drifting loss (default: 0.02,0.05,0.2)",
    )
    parser.add_argument(
        "--drift-neg-max-total",
        type=int,
        default=32,
        help="Maximum negatives passed to drifting loss (default: 32)",
    )
    parser.add_argument(
        "--drift-same-ratio",
        type=float,
        default=0.2,
        help="Ratio of same-seed negatives among history negatives (default: 0.2)",
    )
    parser.add_argument(
        "--drift-history-tail-exclude",
        type=float,
        default=0.05,
        help="Exclude latest tail ratio [0,1) from history sampling (default: 0.05)",
    )
    parser.add_argument(
        "--drift-use-a6-norm",
        action="store_true",
        default=True,
        help="Enable A.6 feature/drift normalization (default: True)",
    )
    parser.add_argument(
        "--no-drift-use-a6-norm",
        action="store_false",
        dest="drift_use_a6_norm",
        help="Disable A.6 feature/drift normalization",
    )
    parser.add_argument(
        "--drift-a6-eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for A.6 normalization (default: 1e-6)",
    )
    parser.add_argument(
        "--drift-hardneg-k",
        type=int,
        default=64,
        help="Hard-negative top-k count (default: 64, 0=disabled)",
    )
    parser.add_argument(
        "--drift-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay for automatic drift scaling (default: 0.99)",
    )
    parser.add_argument(
        "--drift-scale-eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon for drift scaling (default: 1e-8)",
    )

    args = parser.parse_args()
    drift_temperatures = [
        float(item.strip())
        for item in args.drift_temperatures.split(",")
        if item.strip()
    ]
    if not drift_temperatures:
        raise ValueError("drift_temperatures must include at least one value")
    if any(temp <= 0.0 for temp in drift_temperatures):
        raise ValueError("All drift_temperatures must be positive")
    if not (0.0 <= args.drift_same_ratio <= 1.0):
        raise ValueError("drift_same_ratio must be in [0.0, 1.0]")
    if not (0.0 <= args.drift_history_tail_exclude < 1.0):
        raise ValueError("drift_history_tail_exclude must be in [0.0, 1.0)")
    if args.drift_a6_eps <= 0.0:
        raise ValueError("drift_a6_eps must be positive")
    if args.drift_hardneg_k < 0:
        raise ValueError("drift_hardneg_k must be >= 0")
    if not (0.0 <= args.mf_flow_ratio <= 1.0):
        raise ValueError("mf_flow_ratio must be in [0.0, 1.0]")
    if not (0.0 <= args.mf_t1_ratio <= 1.0):
        raise ValueError("mf_t1_ratio must be in [0.0, 1.0]")
    if not (0.0 <= args.mf_r0_ratio <= 1.0):
        raise ValueError("mf_r0_ratio must be in [0.0, 1.0]")
    if args.mf_flow_ratio + args.mf_t1_ratio + args.mf_r0_ratio > 1.0:
        raise ValueError("mf_flow_ratio + mf_t1_ratio + mf_r0_ratio must be <= 1.0")
    if args.mf_onestep_lambda < 0.0:
        raise ValueError("mf_onestep_lambda must be >= 0.0")
    if args.pca_fit_source == "all" and not args.use_drift_loss:
        raise ValueError(
            "--pca-fit-source all requires --use-drift-loss "
            "(history checkpoints are needed for PCA fitting)."
        )

    if args.dataset in {"mnist", "fashion_mnist"} and args.arch != "mlp":
        raise ValueError(
            "MNIST/Fashion-MNIST currently supports --arch mlp only."
        )
    if args.dataset == "cifar10" and args.arch not in {"resnet20", "resnet18"}:
        raise ValueError("CIFAR-10 currently supports --arch resnet20/resnet18 only.")

    # DeepWeightFlow ResNet-18 defaults (applied only when user keeps parser defaults).
    if args.dataset == "cifar10" and args.arch == "resnet18":
        if args.hidden_dim == parser.get_default("hidden_dim"):
            args.hidden_dim = 1024
        if args.time_embed_dim == parser.get_default("time_embed_dim"):
            args.time_embed_dim = 128

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

    # Use RESULTS_DIR environment variable if set
    save_dir = os.environ.get("RESULTS_DIR", args.save_dir)

    return FlowTrainingArgs(
        dataset=args.dataset,
        arch=args.arch,
        method=args.method,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
        dropout=args.dropout,
        sigma=args.sigma,
        source_std=args.source_std,
        t_dist=args.t_dist,
        n_samples=args.n_samples,
        integration_method=args.integration_method,
        checkpoint_dir=checkpoint_dir,
        save_dir=Path(save_dir),
        device=get_available_device(),
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        pca_fit_source=args.pca_fit_source,
        log_freq=args.log_freq,
        save_generated_models=args.save_generated_models,
        mf_flow_ratio=args.mf_flow_ratio,
        mf_t1_ratio=args.mf_t1_ratio,
        mf_r0_ratio=args.mf_r0_ratio,
        mf_onestep_lambda=args.mf_onestep_lambda,
        mf_time_dist=args.mf_time_dist,
        mf_time_dist_mu=args.mf_time_dist_mu,
        mf_time_dist_sigma=args.mf_time_dist_sigma,
        mf_loss_gamma=args.mf_loss_gamma,
        mf_jvp_api=args.mf_jvp_api,
        eval_steps=args.eval_steps,
        use_drift_loss=args.use_drift_loss,
        drift_lambda=args.drift_lambda,
        drift_target_ratio=args.drift_target_ratio,
        drift_auto_scale=args.drift_auto_scale,
        drift_temperatures=drift_temperatures,
        drift_neg_max_total=args.drift_neg_max_total,
        drift_same_ratio=args.drift_same_ratio,
        drift_history_tail_exclude=args.drift_history_tail_exclude,
        drift_use_a6_norm=args.drift_use_a6_norm,
        drift_a6_eps=args.drift_a6_eps,
        drift_hardneg_k=args.drift_hardneg_k,
        drift_ema_decay=args.drift_ema_decay,
        drift_scale_eps=args.drift_scale_eps,
    )
