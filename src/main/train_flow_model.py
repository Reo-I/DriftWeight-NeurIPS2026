"""
Training script for weight generation models (Flow Matching / MeanFlow).

Supports two generative methods:
    - flowmatching: Standard Conditional Flow Matching (DeepWeightFlow)
    - meanflow: MeanFlow with average velocity (few-step generation)

Both methods train on model weights from MNIST/Fashion-MNIST/CIFAR-10
checkpoints.

Usage:
    # Flow Matching (default):
    python -m src.main.train_flow_model --method flowmatching

    # MeanFlow:
    python -m src.main.train_flow_model --method meanflow

References:
    - DeepWeightFlow: https://arxiv.org/pdf/2601.05052
    - MeanFlow: https://arxiv.org/pdf/2505.13447
"""

import json
import random

import numpy as np
import torch

from src.cli import FlowTrainingArgs, parse_flow_args
from src.data import get_cifar10_loaders, get_fashion_mnist_loaders, get_mnist_loaders
from src.flowmatching import FlowMatching, WeightSpaceFlowModel
from src.flowmatching.bn_recalibration import has_batchnorm, recalibrate_bn_stats
from src.flowmatching.mean_flow import WeightSpaceMeanFlow
from src.flowmatching.meanflow_model import WeightSpaceMeanFlowModel
from src.flowmatching.pca import WeightSpacePCA
from src.flowmatching.weight_dataset import (
    create_combined_weight_loader,
    create_flow_dataloaders,
    create_history_weight_loader,
    load_weight_history_from_checkpoints,
    load_weight_history_metadata,
    load_weights_from_checkpoints,
)
from src.flowmatching.weight_space import WeightSpaceObjectStateDict
from src.models import MLP, ResNet18, ResNet20
from src.utils import log_device_info, logger

# Set random seeds for reproducibility (same as DeepWeightFlow paper)
RANDOM_SEED = 42


def _mlp_hidden_dim_for_dataset(dataset: str) -> int:
    """Resolve MLP width from dataset (DeepWeightFlow-compatible presets)."""
    if dataset == "mnist":
        return 32
    if dataset == "fashion_mnist":
        return 128
    raise ValueError(f"Unsupported MLP dataset: {dataset}")


def _build_target_model(args: FlowTrainingArgs) -> torch.nn.Module:
    """Build target architecture model from CLI args."""
    if args.dataset in {"mnist", "fashion_mnist"} and args.arch == "mlp":
        return MLP(hidden_dim=_mlp_hidden_dim_for_dataset(args.dataset))
    if args.dataset == "cifar10" and args.arch == "resnet20":
        return ResNet20()
    if args.dataset == "cifar10" and args.arch == "resnet18":
        return ResNet18()
    raise ValueError(f"Unsupported dataset/arch combination: {args.dataset}/{args.arch}")


def evaluate_generated_weights(
    generated_weights: torch.Tensor,
    args: FlowTrainingArgs,
    model_builder,
    test_loader: torch.utils.data.DataLoader,
    calibration_loader: torch.utils.data.DataLoader | None = None,
) -> dict:
    """Evaluate generated model weights on dataset test split.

    Args:
        generated_weights: Generated weight tensor of shape (n_samples, dim).
        args: Training arguments.
        model_builder: Callable that returns a fresh target model.
        test_loader: MNIST test data loader.
        calibration_loader: Optional loader for BN recalibration.

    Returns:
        Dictionary containing evaluation results.
    """
    # Save generated weights if requested
    if args.save_generated_models:
        generated_models_dir = args.save_dir / "generated_models"
        generated_models_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.n_samples):
            model_path = generated_models_dir / f"generated_model_{i:04d}.pt"
            torch.save(generated_weights[i].cpu(), model_path)

        logger.info(
            f"Saved {args.n_samples} generated models to {generated_models_dir}"
        )

    # Evaluate each generated model
    accuracies = []
    device = args.device

    for i in range(args.n_samples):
        model = model_builder()
        wso = WeightSpaceObjectStateDict.from_flat(
            generated_weights[i], model, device=device
        )
        wso.apply_to_model(model)
        model.to(device)

        if calibration_loader is not None and has_batchnorm(model):
            model = recalibrate_bn_stats(
                model,
                calibration_loader=calibration_loader,
                device=device,
            )

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        accuracies.append(accuracy)

        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{args.n_samples} models")

    accuracies_np = np.array(accuracies)
    results = {
        "method": args.method,
        "n_samples": args.n_samples,
        "accuracies": accuracies,
        "mean_accuracy": float(accuracies_np.mean()),
        "std_accuracy": float(accuracies_np.std()),
        "min_accuracy": float(accuracies_np.min()),
        "max_accuracy": float(accuracies_np.max()),
        "source_std": args.source_std,
        "use_pca": args.use_pca,
        "pca_components": args.pca_components if args.use_pca else None,
    }

    # Add method-specific info
    if args.method == "flowmatching":
        results["integration_method"] = args.integration_method

    logger.info("=" * 60)
    logger.info("Generation Results:")
    logger.info(f"  Method: {args.method}")
    logger.info(f"  Models evaluated: {len(accuracies)}")
    logger.info(
        f"  Mean Accuracy: {results['mean_accuracy']:.2f}% "
        f"\u00b1 {results['std_accuracy']:.2f}%"
    )
    logger.info(f"  Min Accuracy: {results['min_accuracy']:.2f}%")
    logger.info(f"  Max Accuracy: {results['max_accuracy']:.2f}%")
    if args.use_pca:
        logger.info(f"  PCA Components: {args.pca_components}")
    logger.info("=" * 60)

    return results


def generate_and_evaluate(
    trainer: FlowMatching | WeightSpaceMeanFlow,
    args: FlowTrainingArgs,
    model_builder,
    test_loader: torch.utils.data.DataLoader,
    pca: WeightSpacePCA | None = None,
    n_steps: int | None = None,
    calibration_loader: torch.utils.data.DataLoader | None = None,
) -> dict:
    """Generate weights and evaluate them on the test set.

    Args:
        trainer: Trained FlowMatching or MeanFlow instance.
        args: Training arguments.
        model_builder: Callable that returns a fresh target model.
        test_loader: MNIST test data loader.
        pca: Optional fitted PCA object for inverse transform.
        n_steps: Number of generation steps. If None, uses default from args.
        calibration_loader: Optional loader for BN recalibration.

    Returns:
        Dictionary containing evaluation results.
    """
    # Determine step count
    if n_steps is None:
        n_steps = args.eval_steps[0]

    logger.info(
        f"Generating {args.n_samples} new models "
        f"with {n_steps} steps for evaluation..."
    )

    # Generate weights (method-specific)
    if args.method == "meanflow":
        assert isinstance(trainer, WeightSpaceMeanFlow)
        generated_latent = trainer.generate_weights(
            n_samples=args.n_samples,
            source_std=args.source_std,
            n_steps=n_steps,
        )
    else:
        assert isinstance(trainer, FlowMatching)
        generated_latent = trainer.generate_weights(
            n_samples=args.n_samples,
            source_std=args.source_std,
            n_steps=n_steps,
            method=args.integration_method,
        )

    # Apply inverse PCA transform if PCA was used
    if pca is not None:
        logger.info("Applying PCA inverse transform to generated weights...")
        generated_weights = pca.inverse_transform(
            generated_latent, device=args.device
        )
    else:
        generated_weights = generated_latent

    results = evaluate_generated_weights(
        generated_weights,
        args,
        model_builder,
        test_loader,
        calibration_loader=calibration_loader,
    )
    results["n_steps"] = n_steps
    return results


def _build_config_dict(
        args: FlowTrainingArgs,
        weight_dim: int,
        actual_dim: int,
        pca: WeightSpacePCA | None,
        weight_objects: list,
    ) -> dict:
    """Build configuration dictionary for saving.

    Args:
        args: Training arguments.
        weight_dim: Original weight dimension.
        actual_dim: Actual dimension (after PCA).
        pca: Optional PCA object.
        weight_objects: List of loaded weight objects.

    Returns:
        Configuration dictionary.
    """
    config_dict = {
        "method": args.method,
        "n_iters": args.n_iters,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "time_embed_dim": args.time_embed_dim,
        "dropout": args.dropout,
        "sigma": args.sigma,
        "source_std": args.source_std,
        "n_samples": args.n_samples,
        "dataset": args.dataset,
        "arch": args.arch,
        "weight_dim": weight_dim,
        "actual_dim": actual_dim,
        "use_pca": args.use_pca,
        "pca_components": args.pca_components if args.use_pca else None,
        "pca_fit_source": args.pca_fit_source if args.use_pca else None,
        "pca_variance_captured": pca.total_variance_captured if pca else None,
        "num_training_models": len(weight_objects),
    }

    # Shared evaluation config
    config_dict["eval_steps"] = args.eval_steps

    if args.method == "flowmatching":
        config_dict.update(
            {
                "t_dist": args.t_dist,
                "integration_method": args.integration_method,
            }
        )
    else:
        config_dict.update(
            {
                "mf_flow_ratio": args.mf_flow_ratio,
                "mf_t1_ratio": args.mf_t1_ratio,
                "mf_r0_ratio": args.mf_r0_ratio,
                "mf_onestep_lambda": args.mf_onestep_lambda,
                "mf_time_dist": args.mf_time_dist,
                "mf_time_dist_mu": args.mf_time_dist_mu,
                "mf_time_dist_sigma": args.mf_time_dist_sigma,
                "mf_loss_gamma": args.mf_loss_gamma,
                "mf_jvp_api": args.mf_jvp_api,
                "use_drift_loss": args.use_drift_loss,
                "drift_lambda": args.drift_lambda,
                "drift_target_ratio": args.drift_target_ratio,
                "drift_auto_scale": args.drift_auto_scale,
                "drift_temperatures": args.drift_temperatures,
                "drift_neg_max_total": args.drift_neg_max_total,
                "drift_same_ratio": args.drift_same_ratio,
                "drift_history_tail_exclude": args.drift_history_tail_exclude,
                "drift_use_a6_norm": args.drift_use_a6_norm,
                "drift_a6_eps": args.drift_a6_eps,
                "drift_hardneg_k": args.drift_hardneg_k,
                "drift_ema_decay": args.drift_ema_decay,
                "drift_scale_eps": args.drift_scale_eps,
            }
        )

    return config_dict


def _train_flowmatching(
        args: FlowTrainingArgs,
        target_loader: torch.utils.data.DataLoader,
        actual_dim: int,
    ) -> FlowMatching:
    """Build and train a Flow Matching model.

    Args:
        args: Training arguments.
        target_loader: Target distribution DataLoader.
        actual_dim: Input dimension for the model.

    Returns:
        Trained FlowMatching instance.
    """
    flow_model = WeightSpaceFlowModel(
        input_dim=actual_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        time_embed_dim=args.time_embed_dim,
    ).to(args.device)

    num_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    logger.info(f"Flow model parameters: {num_params:,}")

    fm = FlowMatching(
        targetloader=target_loader,
        model=flow_model,
        mode="velocity",
        t_dist=args.t_dist,
        device=args.device,
        source_std=args.source_std,
        sigma=args.sigma,
    )
    fm.input_dim = actual_dim

    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.n_iters,
        eta_min=1e-6,
    )

    logger.info("Starting Flow Matching training...")
    logger.info(f"  Iterations: {args.n_iters}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Time dist: {args.t_dist}")
    logger.info(f"  PCA: {'Yes' if args.use_pca else 'No'}")

    fm.train(
        n_iters=args.n_iters,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=100,
        log_freq=args.log_freq,
    )

    return fm


def _train_meanflow(
    args: FlowTrainingArgs,
    target_loader: torch.utils.data.DataLoader,
    actual_dim: int,
    history_by_seed: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> WeightSpaceMeanFlow:
    """Build and train a MeanFlow model.

    MeanFlow generates fresh noise internally each iteration.

    Args:
        args: Training arguments.
        target_loader: Target distribution DataLoader.
        actual_dim: Input dimension for the model.

    Returns:
        Trained WeightSpaceMeanFlow instance.
    """
    meanflow_model = WeightSpaceMeanFlowModel(
        input_dim=actual_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        time_embed_dim=args.time_embed_dim,
    ).to(args.device)

    num_params = sum(
        p.numel() for p in meanflow_model.parameters() if p.requires_grad
    )
    logger.info(f"MeanFlow model parameters: {num_params:,}")

    mf = WeightSpaceMeanFlow(
        targetloader=target_loader,
        model=meanflow_model,
        history_by_seed=history_by_seed,
        device=args.device,
        source_std=args.source_std,
        sigma=args.sigma,
        flow_ratio=args.mf_flow_ratio,
        t1_ratio=args.mf_t1_ratio,
        r0_ratio=args.mf_r0_ratio,
        onestep_lambda=args.mf_onestep_lambda,
        time_dist=args.mf_time_dist,
        time_dist_mu=args.mf_time_dist_mu,
        time_dist_sigma=args.mf_time_dist_sigma,
        loss_gamma=args.mf_loss_gamma,
        jvp_api=args.mf_jvp_api,
        use_drift_loss=args.use_drift_loss,
        drift_lambda=args.drift_lambda,
        drift_target_ratio=args.drift_target_ratio,
        drift_auto_scale=args.drift_auto_scale,
        drift_temperatures=args.drift_temperatures,
        drift_neg_max_total=args.drift_neg_max_total,
        drift_same_ratio=args.drift_same_ratio,
        drift_history_tail_exclude=args.drift_history_tail_exclude,
        drift_use_a6_norm=args.drift_use_a6_norm,
        drift_a6_eps=args.drift_a6_eps,
        drift_hardneg_k=args.drift_hardneg_k,
        drift_ema_decay=args.drift_ema_decay,
        drift_scale_eps=args.drift_scale_eps,
    )
    mf.input_dim = actual_dim

    optimizer = torch.optim.AdamW(
        mf.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.n_iters,
        eta_min=1e-6,
    )

    logger.info("Starting MeanFlow training...")
    logger.info(f"  Iterations: {args.n_iters}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Flow ratio (r=t): {args.mf_flow_ratio}")
    logger.info(f"  t=1 ratio: {args.mf_t1_ratio}")
    logger.info(f"  r=0 ratio: {args.mf_r0_ratio}")
    logger.info(f"  1-step aux: consistency (lambda={args.mf_onestep_lambda})")
    logger.info(f"  Time dist: {args.mf_time_dist}")
    logger.info(
        f"  Time dist params: mu={args.mf_time_dist_mu}, "
        f"sigma={args.mf_time_dist_sigma}"
    )
    logger.info(f"  Loss gamma: {args.mf_loss_gamma}")
    logger.info(f"  JVP API: {args.mf_jvp_api}")
    logger.info(f"  Drift loss: {'On' if args.use_drift_loss else 'Off'}")
    if args.use_drift_loss:
        logger.info(
            f"  Drift config: lambda={args.drift_lambda}, "
            f"target_ratio={args.drift_target_ratio}, "
            f"auto_scale={args.drift_auto_scale}, "
            f"temperatures={args.drift_temperatures}, "
            f"neg_max_total={args.drift_neg_max_total}, "
            f"same_ratio={args.drift_same_ratio}, "
            f"tail_exclude={args.drift_history_tail_exclude}, "
            f"use_a6_norm={args.drift_use_a6_norm}, "
            f"a6_eps={args.drift_a6_eps}, "
            f"hardneg_k={args.drift_hardneg_k}"
        )
    logger.info(f"  Eval steps: {args.eval_steps}")
    logger.info(f"  PCA: {'Yes' if args.use_pca else 'No'}")

    mf.train(
        n_iters=args.n_iters,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=100,
        log_freq=args.log_freq,
    )

    return mf


def main() -> None:
    """Main entry point for flow model training."""
    args = parse_flow_args()

    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {RANDOM_SEED} for reproducibility")

    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.save_dir}")
    logger.info(f"Method: {args.method}")
    # Log GPU info
    log_device_info()

    model_builder = lambda: _build_target_model(args)
    prototype_model = model_builder()
    weight_dim = sum(p.numel() for p in prototype_model.parameters())
    logger.info(f"Target setup: dataset={args.dataset}, arch={args.arch}")
    logger.info(f"Weight space dimension: {weight_dim:,}")

    # Log PCA configuration
    if args.use_pca:
        logger.info(f"PCA enabled with {args.pca_components} components")
        if args.dataset == "cifar10" and args.arch == "resnet18":
            logger.info(
                "ResNet-18 follows the DeepWeightFlow setup with standard "
                "IncrementalPCA-reduced weights."
            )
            if args.pca_components not in {None, 99}:
                logger.warning(
                    "DeepWeightFlow ResNet-18 uses 99 PCA components for 100 "
                    "models. Current explicit setting is %s.",
                    args.pca_components,
                )
        logger.info(f"Dimension reduction: {weight_dim:,} -> {args.pca_components}")
    else:
        if args.dataset == "cifar10" and args.arch == "resnet18":
            logger.warning(
                "PCA is disabled, but DeepWeightFlow ResNet-18 uses standard "
                "IncrementalPCA-reduced weights. This diverges from the paper setup."
            )
        else:
            logger.warning("PCA is disabled - using full weight space")

    # Load model weights and build target dataloader (method-specific)
    logger.info(f"Loading weights from {args.checkpoint_dir}...")
    history_by_seed: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    if args.method == "meanflow":
        # MeanFlow target data is final.pt only (same as flowmatching).
        weight_objects, seed_ids = load_weights_from_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            model_builder=model_builder,
            device=args.device,
        )
        logger.info("Loaded final checkpoints for MeanFlow target: %d samples", len(weight_objects))

        # cifar10+resnet18 has ~11M params x ~3900 history checkpoints (~166GB),
        # which cannot fit in memory.  Use streaming PCA for this combination;
        # all other dataset/arch pairs are small enough for bulk loading.
        use_streaming = (args.dataset == "cifar10" and args.arch == "resnet18")

        history_metadata = None
        history_objects = None
        history_seed_ids: list[int] = []
        history_iter_positions: list[float] = []
        history_bank: torch.Tensor | None = None

        if args.use_drift_loss:
            if use_streaming:
                history_metadata = load_weight_history_metadata(
                    checkpoint_dir=args.checkpoint_dir,
                    model_builder=model_builder,
                )
                logger.info(
                    "History metadata collected for drift negatives: %d checkpoints "
                    "[streaming mode]",
                    len(history_metadata),
                )
            else:
                history_objects, history_seed_ids, history_iter_positions = (
                    load_weight_history_from_checkpoints(
                        checkpoint_dir=args.checkpoint_dir,
                        model_builder=model_builder,
                        device=args.device,
                    )
                )
                logger.info(
                    "Loaded history checkpoints for drift negatives: %d samples "
                    "[bulk mode]",
                    len(history_objects),
                )
                history_bank = torch.stack(
                    [wso.flatten(device=args.device) for wso in history_objects]
                )

        # --- PCA fitting ---
        if (
            args.use_pca
            and args.pca_fit_source == "all"
            and use_streaming
            and history_metadata is not None
        ):
            # Streaming path: fit PCA via GPU Gram-matrix decomposition
            final_flat = torch.stack(
                [wso.flatten(device="cpu") for wso in weight_objects]
            )
            n_final = final_flat.size(0)
            n_history = len(history_metadata)
            n_total_fit = n_final + n_history
            logger.info(
                "PCA fit source: all checkpoints (final=%d + history=%d = %d) "
                "[streaming]",
                n_final, n_history, n_total_fit,
            )

            combined_loader = create_combined_weight_loader(
                final_weights=final_flat,
                history_metadata=history_metadata,
                model_builder=model_builder,
                device="cpu",
            )

            max_valid_components = min(n_total_fit - 1, weight_dim)
            pca_components = args.pca_components
            if pca_components is None:
                pca_components = max_valid_components
            elif pca_components > max_valid_components:
                logger.warning(
                    "Requested pca_components=%d exceeds max valid=%d. "
                    "Auto-adjusting to %d.",
                    pca_components, max_valid_components, max_valid_components,
                )
                pca_components = max_valid_components

            pca = WeightSpacePCA(n_components=pca_components)
            streaming_batch = max(200, pca_components)
            pca.fit_streaming(
                weight_loader=combined_loader,
                n_total=n_total_fit,
                n_features=weight_dim,
                batch_size=streaming_batch,
            )

            target_tensor = pca.transform(final_flat)
            actual_dim = pca_components
            del final_flat

            logger.info("PCA variance captured: %.4f%%", pca.total_variance_captured * 100)

            from torch.utils.data import DataLoader, TensorDataset

            target_tensor_dev = target_tensor.to(args.device)
            seed_tensor = torch.tensor(seed_ids, device=args.device, dtype=torch.long)
            target_loader = DataLoader(
                TensorDataset(target_tensor_dev, seed_tensor),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
            )
            del target_tensor, target_tensor_dev

        elif args.use_pca and args.pca_fit_source == "all" and history_bank is not None:
            # Bulk path: fit PCA on final + history (all in memory)
            final_flat = torch.stack(
                [wso.flatten(device=args.device) for wso in weight_objects]
            )
            pca_fit_weights = torch.cat([final_flat, history_bank], dim=0)
            logger.info(
                "PCA fit source: all checkpoints (final=%d + history=%d = %d) "
                "[bulk]",
                final_flat.size(0), history_bank.size(0), pca_fit_weights.size(0),
            )
            del final_flat

            target_loader, actual_dim, pca = create_flow_dataloaders(
                weight_objects=weight_objects,
                seed_ids=seed_ids,
                batch_size=args.batch_size,
                device=args.device,
                use_pca=True,
                pca_components=args.pca_components,
                pca_fit_weights=pca_fit_weights,
            )
            del pca_fit_weights

        elif args.use_pca:
            pca_components = args.pca_components
            target_loader, actual_dim, pca = create_flow_dataloaders(
                weight_objects=weight_objects,
                seed_ids=seed_ids,
                batch_size=args.batch_size,
                device=args.device,
                use_pca=True,
                pca_components=pca_components,
            )
            logger.info(
                "MeanFlow PCA fit source: final checkpoints only (n=%d)",
                len(weight_objects),
            )
        else:
            pca = None
            target_loader, actual_dim, _ = create_flow_dataloaders(
                weight_objects=weight_objects,
                seed_ids=seed_ids,
                batch_size=args.batch_size,
                device=args.device,
                use_pca=False,
            )

        # --- Build history_by_seed for drift loss ---
        if args.use_drift_loss and use_streaming and history_metadata is not None:
            # Streaming: transform history via batched disk reads
            n_hist = len(history_metadata)
            history_loader = create_history_weight_loader(
                history_metadata, model_builder, device="cpu",
            )
            if pca is not None:
                history_bank = pca.transform_streaming(
                    weight_loader=history_loader,
                    n_total=n_hist,
                    batch_size=max(200, pca.n_components),
                    device=args.device,
                )
            else:
                history_bank = history_loader(0, n_hist).to(args.device)

            history_seed_ids_tensor = torch.tensor(
                history_metadata.seed_ids, device=args.device, dtype=torch.long,
            )
            history_iter_positions_tensor = torch.tensor(
                history_metadata.iter_positions, device=args.device, dtype=torch.float32,
            )
            history_by_seed = {}
            for seed_val in torch.unique(history_seed_ids_tensor).tolist():
                seed_val_int = int(seed_val)
                seed_mask = history_seed_ids_tensor == seed_val_int
                history_by_seed[seed_val_int] = (
                    history_bank[seed_mask],
                    history_iter_positions_tensor[seed_mask],
                )
            del history_bank

        elif args.use_drift_loss and history_bank is not None:
            # Bulk: history already in memory, just PCA-transform if needed
            if pca is not None:
                history_bank = pca.transform(history_bank)
            history_seed_ids_tensor = torch.tensor(
                history_seed_ids, device=args.device, dtype=torch.long,
            )
            history_iter_positions_tensor = torch.tensor(
                history_iter_positions, device=args.device, dtype=torch.float32,
            )
            history_by_seed = {}
            for seed_val in torch.unique(history_seed_ids_tensor).tolist():
                seed_val_int = int(seed_val)
                seed_mask = history_seed_ids_tensor == seed_val_int
                history_by_seed[seed_val_int] = (
                    history_bank[seed_mask],
                    history_iter_positions_tensor[seed_mask],
                )
            del history_bank

    else:
        weight_objects, seed_ids = load_weights_from_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            model_builder=model_builder,
            device=args.device,
        )
        logger.info(f"Loaded {len(weight_objects)} model weights")
        target_loader, actual_dim, pca = create_flow_dataloaders(
            weight_objects=weight_objects,
            seed_ids=seed_ids,
            batch_size=args.batch_size,
            device=args.device,
            use_pca=args.use_pca,
            pca_components=args.pca_components,
        )
    logger.info(f"Input dimension: {actual_dim}")

    # Save PCA if used
    if pca is not None:
        pca_path = args.save_dir / "pca.npz"
        pca.save(str(pca_path))
        logger.info(f"PCA saved to {pca_path}")

    # Train model (method-specific)
    if args.method == "meanflow":
        trainer = _train_meanflow(
            args,
            target_loader,
            actual_dim,
            history_by_seed=history_by_seed,
        )
    else:
        trainer = _train_flowmatching(
            args, target_loader, actual_dim
        )

    # Save trained model
    model_path = args.save_dir / "flow_model.pt"
    trainer.save_checkpoint(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Load evaluation (and optional calibration) data
    calibration_loader = None
    if args.dataset == "mnist":
        logger.info("Loading MNIST test data for evaluation...")
        _, test_loader = get_mnist_loaders(batch_size=64)
    elif args.dataset == "fashion_mnist":
        logger.info("Loading Fashion-MNIST test data for evaluation...")
        _, test_loader = get_fashion_mnist_loaders(batch_size=64)
    elif args.dataset == "cifar10":
        cifar_root = (
            "data/CIFAR10_resnet18/raw"
            if args.arch == "resnet18"
            else "data/CIFAR10/raw"
        )
        calibration_loader, test_loader = get_cifar10_loaders(
            batch_size=128,
            data_root=cifar_root,
        )
        logger.info(
            "Loading CIFAR-10 data for evaluation and BatchNorm recalibration..."
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Evaluation step sizes
    eval_steps = sorted(args.eval_steps)
    logger.info(f"Evaluation step sizes: {eval_steps}")

    # Generate and evaluate for each step size
    all_results: dict[str, dict] = {}
    for n_steps in eval_steps:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating with {n_steps} step(s)...")
        logger.info(f"{'=' * 60}")
        results = generate_and_evaluate(
            trainer,
            args,
            model_builder,
            test_loader,
            pca,
            n_steps=n_steps,
            calibration_loader=calibration_loader,
        )
        all_results[str(n_steps)] = results

    # Print summary across all step sizes
    logger.info("\n" + "=" * 60)
    logger.info("Summary across all step sizes:")
    logger.info(f"{'Steps':>6s} | {'Mean Acc':>10s} | {'Std':>8s} | {'Min':>8s} | {'Max':>8s}")
    logger.info("-" * 60)
    for n_steps_str, res in all_results.items():
        logger.info(
            f"{n_steps_str:>6s} | "
            f"{res['mean_accuracy']:>9.2f}% | "
            f"{res['std_accuracy']:>7.2f}% | "
            f"{res['min_accuracy']:>7.2f}% | "
            f"{res['max_accuracy']:>7.2f}%"
        )
    logger.info("=" * 60)

    # Save evaluation results as JSON
    results_path = args.save_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")

    # Save training config
    config_dict = _build_config_dict(
        args, weight_dim, actual_dim, pca, weight_objects
    )
    config_path = args.save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    main()
