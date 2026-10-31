"""Batch Normalization statistics recalibration for generated weights.

When generating neural network weights using Flow Matching, the BatchNorm
running statistics (running_mean, running_var) may not be valid for the
newly generated weights. This module provides utilities to recalibrate
these statistics by running a forward pass on calibration data.

Note: For MLP models without BatchNorm (like the MNIST MLP), this module
is not needed. It is provided for completeness when working with models
that include BatchNorm layers (e.g., ResNet, CNN).

Reference: DeepWeightFlow paper Section 3.3 and Appendix D.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import logger


def recalibrate_bn_stats(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: str | torch.device = "cuda",
    verbose: bool = False,
) -> nn.Module:
    """Recalibrate BatchNorm running statistics for a model.

    Resets the running statistics of all BatchNorm layers and recalculates
    them by performing forward passes on calibration data. This is necessary
    when model weights have been generated or modified, as the original
    BatchNorm statistics may no longer be valid.

    The recalibration uses cumulative moving average (momentum=None) to
    compute exact statistics over the calibration dataset.

    Args:
        model: PyTorch model with BatchNorm layers.
        calibration_loader: DataLoader with calibration data.
        device: Device to perform computation on.
        verbose: If True, print statistics for each BatchNorm layer.

    Returns:
        Model with recalibrated BatchNorm statistics.

    Example:
        >>> model = generated_resnet
        >>> calibration_loader = get_cifar10_loader(batch_size=128)
        >>> model = recalibrate_bn_stats(model, calibration_loader)
    """
    model.train()
    model.to(device)

    # Find all BatchNorm layers
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append((name, module))

    if not bn_layers:
        logger.debug("No BatchNorm layers found, skipping recalibration")
        return model

    logger.info(f"Recalibrating {len(bn_layers)} BatchNorm layers")

    # Reset running statistics and set momentum to None (cumulative average)
    for name, bn in bn_layers:
        bn.reset_running_stats()
        bn.momentum = None

    # Forward pass through calibration data to compute statistics
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device)
            _ = model(inputs)

    # Print statistics if verbose
    if verbose:
        for name, bn in bn_layers:
            mean_val = bn.running_mean.mean().item()
            var_val = bn.running_var.mean().item()
            num_batches = bn.num_batches_tracked.item()
            logger.info(
                f"{name}: mean={mean_val:.4f}, var={var_val:.4f}, "
                f"batches_tracked={num_batches}"
            )

    # Reset momentum to default value
    for name, bn in bn_layers:
        bn.momentum = 0.1

    logger.info("BatchNorm recalibration complete")
    return model


def has_batchnorm(model: nn.Module) -> bool:
    """Check if a model contains BatchNorm layers.

    Args:
        model: PyTorch model to check.

    Returns:
        True if model contains any BatchNorm layers.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False
