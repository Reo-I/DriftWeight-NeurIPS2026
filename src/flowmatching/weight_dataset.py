"""Dataset utilities for loading trained model weights for Flow Matching.

This module provides functions to load checkpoints from trained models
and create PyTorch DataLoaders for use in Flow Matching training.
Includes support for PCA dimensionality reduction.

Reference: DeepWeightFlow paper (https://arxiv.org/pdf/2601.05052)
For MNIST MLP, PCA reduces 26,506 -> 99 dimensions.
"""

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.flowmatching.pca import WeightSpacePCA
from src.flowmatching.weight_space import WeightSpaceObjectStateDict
from src.utils.logger import logger


def load_weights_from_checkpoints(
    checkpoint_dir: Path,
    model_builder: Callable[[], torch.nn.Module],
    num_models: int | None = None,
    device: str = "cpu",
) -> tuple[list[WeightSpaceObjectStateDict], list[int]]:
    """Load model weights from checkpoint files.

    Scans the checkpoint directory for model weight files and loads them
    as architecture-agnostic WeightSpaceObjectStateDict instances.

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories.
        model_builder: Zero-argument function that builds the target model.
        num_models: Maximum number of models to load (None = load all).
        device: Device to load weights to.

    Returns:
        Tuple of:
            - List of WeightSpaceObjectStateDict objects
            - List of integer seed IDs aligned with loaded weights

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist.
        ValueError: If no checkpoints are found.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    weight_objects = []
    seed_ids: list[int] = []
    seed_dirs = sorted(checkpoint_dir.glob("seed*"))

    if not seed_dirs:
        raise ValueError(f"No seed directories found in {checkpoint_dir}")

    logger.info(f"Found {len(seed_dirs)} seed directories in {checkpoint_dir}")

    for seed_dir in seed_dirs:
        if num_models is not None and len(weight_objects) >= num_models:
            break

        # Look for final weights file
        final_weights = list(seed_dir.glob("final.pt"))
        if final_weights:
            weight_file = final_weights[0]
            logger.info(f"Loaded weights from {weight_file}")
        else:
            # Fall back to latest checkpoint
            checkpoints = sorted(seed_dir.glob("checkpoint_iter*.pt"))
            if not checkpoints:
                logger.warning(f"No checkpoints found in {seed_dir}")
                continue
            weight_file = checkpoints[-1]

        try:
            seed_name = seed_dir.name
            if not seed_name.startswith("seed"):
                logger.warning(f"Unexpected seed directory name: {seed_name}")
                continue
            seed_id = int(seed_name[4:])

            # Load state dict
            state_dict = torch.load(weight_file, map_location=device)

            # Handle different checkpoint formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            model = model_builder()
            model.load_state_dict(state_dict)

            # Convert to generic weight-space object
            wso = WeightSpaceObjectStateDict.from_model(model)
            weight_objects.append(wso)
            seed_ids.append(seed_id)

            logger.debug(f"Loaded weights from {weight_file}")

        except Exception as e:
            logger.warning(f"Failed to load {weight_file}: {e}")
            continue

    if not weight_objects:
        raise ValueError(f"No valid weights loaded from {checkpoint_dir}")

    logger.info(f"Loaded {len(weight_objects)} model weights")
    return weight_objects, seed_ids


def load_weight_history_from_checkpoints(
    checkpoint_dir: Path,
    model_builder: Callable[[], torch.nn.Module],
    num_seeds: int | None = None,
    device: str = "cpu",
) -> tuple[list[WeightSpaceObjectStateDict], list[int], list[float]]:
    """Load per-seed checkpoint history for MeanFlow drifting loss.

    For each ``seedXXXX`` directory, this loader collects
    ``checkpoint_iter*.pt`` files only (final.pt is excluded), and drops
    the last checkpoint_iter sample per seed to avoid duplicating final state.
    It returns an aligned tuple of:
      - architecture-agnostic weight objects
      - seed IDs
      - normalized iteration position in [0, 1] within the same seed

    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories.
        model_builder: Zero-argument function that builds the target model.
        num_seeds: Maximum number of seed directories to load.
        device: Device to load weights to.

    Returns:
        Tuple of (weight_objects, seed_ids, iter_positions).
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    def _iter_from_name(path: Path) -> int:
        stem = path.stem  # e.g. checkpoint_iter1200
        if "checkpoint_iter" in stem:
            try:
                return int(stem.split("checkpoint_iter", maxsplit=1)[1])
            except Exception:
                return -1
        return 10**12  # keep final.pt at the end

    weight_objects: list[WeightSpaceObjectStateDict] = []
    seed_ids: list[int] = []
    iter_positions: list[float] = []
    seed_dirs = sorted(checkpoint_dir.glob("seed*"))

    if not seed_dirs:
        raise ValueError(f"No seed directories found in {checkpoint_dir}")

    logger.info(f"Found {len(seed_dirs)} seed directories in {checkpoint_dir}")

    loaded_seed_count = 0
    for seed_dir in seed_dirs:
        if num_seeds is not None and loaded_seed_count >= num_seeds:
            break

        seed_name = seed_dir.name
        if not seed_name.startswith("seed"):
            logger.warning(f"Unexpected seed directory name: {seed_name}")
            continue

        try:
            seed_id = int(seed_name[4:])
        except Exception:
            logger.warning(f"Failed to parse seed id from {seed_name}")
            continue

        ckpt_files = sorted(seed_dir.glob("checkpoint_iter*.pt"), key=_iter_from_name)
        if not ckpt_files:
            logger.warning(f"No checkpoints found in {seed_dir}")
            continue
        # Drop the latest checkpoint_iter entry per seed, which is typically
        # equivalent to final.pt and would duplicate target finals in history.
        if len(ckpt_files) == 1:
            logger.warning(
                "Only one checkpoint_iter found in %s; skipping history for this seed.",
                seed_dir,
            )
            continue
        ckpt_files = ckpt_files[:-1]

        n_ckpts = len(ckpt_files)
        for idx, weight_file in enumerate(ckpt_files):
            try:
                state_dict = torch.load(weight_file, map_location=device)
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]

                model = model_builder()
                model.load_state_dict(state_dict)
                wso = WeightSpaceObjectStateDict.from_model(model)

                weight_objects.append(wso)
                seed_ids.append(seed_id)
                if n_ckpts <= 1:
                    iter_positions.append(1.0)
                else:
                    iter_positions.append(float(idx) / float(n_ckpts - 1))
            except Exception as e:
                logger.warning(f"Failed to load {weight_file}: {e}")
                continue

        loaded_seed_count += 1

    if not weight_objects:
        raise ValueError(f"No valid checkpoint history loaded from {checkpoint_dir}")

    logger.info(
        "Loaded %d historical checkpoints across %d seeds",
        len(weight_objects),
        loaded_seed_count,
    )
    return weight_objects, seed_ids, iter_positions


def create_flow_dataloaders(
        weight_objects: list[WeightSpaceObjectStateDict],
        seed_ids: list[int],
        batch_size: int = 8,
        device: str = "cpu",
        use_pca: bool = False,
        pca_components: int | None = None,
        pca_fit_weights: torch.Tensor | None = None,
    ) -> tuple[DataLoader, int, WeightSpacePCA | None]:
    """Create target DataLoader for Flow Matching / MeanFlow.

    When use_pca=True, applies PCA dimensionality reduction to the weight
    space. This is crucial for achieving good generation quality - the
    DeepWeightFlow paper uses 99 components for 100 training models.

    Args:
        weight_objects: List of WeightSpaceObjectStateDict to use as targets.
        seed_ids: Seed IDs aligned with weight_objects.
        batch_size: Batch size for dataloaders.
        device: Device for tensors.
        use_pca: Whether to apply PCA dimensionality reduction.
        pca_components: Number of PCA components. If None and use_pca=True,
                        uses n_models - 1 (max possible).
        pca_fit_weights: Pre-flattened weight tensor to fit PCA on.
                         If None, PCA is fit on target weights (default).
                         Useful for fitting PCA on a larger set (e.g. final + history).

    Returns:
        Tuple of (target_loader, actual_dim, pca_object).
        If use_pca=False, pca_object will be None.
        actual_dim is the dimension of the flow matching space
        (original weight dim without PCA, or n_components with PCA).
    """
    # Stack all flattened weights as target distribution
    target_weights = torch.stack([wso.flatten(device=device) for wso in weight_objects])
    n_models, weight_dim = target_weights.shape
    if len(seed_ids) != n_models:
        raise ValueError(
            f"seed_ids length {len(seed_ids)} must match models {n_models}"
        )

    logger.info(f"Target weights shape: {target_weights.shape}")
    logger.info(f"Original weight space dimension: {weight_dim:,}")

    # Apply PCA if requested
    pca = None
    if use_pca:
        fit_source = target_weights if pca_fit_weights is None else pca_fit_weights
        if pca_fit_weights is not None and pca_fit_weights.size(1) != weight_dim:
            raise ValueError(
                "pca_fit_weights feature dimension must match target weights. "
                f"Got fit_dim={pca_fit_weights.size(1)}, target_dim={weight_dim}."
            )
        n_fit = fit_source.size(0)
        max_valid_components = min(n_fit - 1, weight_dim)
        if max_valid_components < 1:
            raise ValueError(
                "PCA requires at least 2 models to define components. "
                f"Got n_fit={n_fit}."
            )

        if pca_components is None:
            pca_components = max_valid_components
        elif pca_components > max_valid_components:
            logger.warning(
                "Requested pca_components=%d exceeds max valid=%d. "
                "Auto-adjusting to %d.",
                pca_components,
                max_valid_components,
                max_valid_components,
            )
            pca_components = max_valid_components
        elif pca_components < 1:
            raise ValueError(f"pca_components must be >= 1, got {pca_components}")

        fit_source_label = (
            f"external ({n_fit} samples)" if pca_fit_weights is not None
            else f"target ({n_fit} samples)"
        )
        logger.info(
            "Applying PCA: %s -> %d dimensions (fit source: %s)",
            f"{weight_dim:,}", pca_components, fit_source_label,
        )

        pca = WeightSpacePCA(n_components=pca_components)
        pca.fit(fit_source)
        target_tensor = pca.transform(target_weights)
        actual_dim = pca_components

        logger.info(f"PCA variance captured: {pca.total_variance_captured:.4%}")
    else:
        target_tensor = target_weights
        actual_dim = weight_dim
        logger.info("PCA disabled - using full weight space")

    logger.info(f"Flow matching dimension: {actual_dim:,}")

    # Create datasets
    target_seed_tensor = torch.tensor(seed_ids, device=device, dtype=torch.long)
    target_dataset = TensorDataset(target_tensor, target_seed_tensor)

    # Create dataloader
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    return target_loader, actual_dim, pca


def create_meanflow_history_dataloader(
    weight_objects: list[WeightSpaceObjectStateDict],
    seed_ids: list[int],
    iter_positions: list[float],
    pca_fit_weight_objects: list[WeightSpaceObjectStateDict] | None = None,
    batch_size: int = 8,
    device: str = "cpu",
    use_pca: bool = False,
    pca_components: int | None = None,
) -> tuple[DataLoader, int, WeightSpacePCA | None]:
    """Create MeanFlow target dataloader with seed and iter-position labels."""
    if len(weight_objects) != len(seed_ids) or len(weight_objects) != len(iter_positions):
        raise ValueError(
            "weight_objects, seed_ids, and iter_positions must have the same length."
        )

    target_weights = torch.stack([wso.flatten(device=device) for wso in weight_objects])
    n_models, weight_dim = target_weights.shape
    logger.info(f"MeanFlow historical targets shape: {target_weights.shape}")
    logger.info(f"Original weight space dimension: {weight_dim:,}")

    pca = None
    if use_pca:
        pca_fit_source = "history checkpoints"
        pca_fit_weights = target_weights
        if pca_fit_weight_objects is not None:
            pca_fit_weights = torch.stack(
                [wso.flatten(device=device) for wso in pca_fit_weight_objects]
            )
            pca_fit_source = "final checkpoints only"
            if pca_fit_weights.size(1) != weight_dim:
                raise ValueError(
                    "PCA fit weights and historical weights must have same feature dimension. "
                    f"Got fit_dim={pca_fit_weights.size(1)}, history_dim={weight_dim}."
                )

        fit_models = pca_fit_weights.size(0)
        max_valid_components = min(fit_models - 1, weight_dim)
        if max_valid_components < 1:
            raise ValueError(
                "PCA requires at least 2 models to define components. "
                f"Got n_models={fit_models} in fit source ({pca_fit_source})."
            )
        if pca_components is None:
            pca_components = max_valid_components
        elif pca_components > max_valid_components:
            logger.warning(
                "Requested pca_components=%d exceeds max valid=%d. "
                "Auto-adjusting to %d.",
                pca_components,
                max_valid_components,
                max_valid_components,
            )
            pca_components = max_valid_components
        elif pca_components < 1:
            raise ValueError(f"pca_components must be >= 1, got {pca_components}")

        logger.info(f"Applying PCA: {weight_dim:,} -> {pca_components} dimensions")
        logger.info(
            "MeanFlow PCA fit source: %s (n=%d), transform target: history checkpoints (n=%d)",
            pca_fit_source,
            fit_models,
            n_models,
        )
        pca = WeightSpacePCA(n_components=pca_components)
        pca.fit(pca_fit_weights)
        target_tensor = pca.transform(target_weights)
        actual_dim = pca_components
        logger.info(f"PCA variance captured: {pca.total_variance_captured:.4%}")
    else:
        target_tensor = target_weights
        actual_dim = weight_dim
        logger.info("PCA disabled - using full weight space")

    target_seed_tensor = torch.tensor(seed_ids, device=device, dtype=torch.long)
    iter_pos_tensor = torch.tensor(iter_positions, device=device, dtype=torch.float32)
    target_dataset = TensorDataset(target_tensor, target_seed_tensor, iter_pos_tensor)
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return target_loader, actual_dim, pca


class HistoryMetadata:
    """Lightweight metadata for checkpoint history (no weight data in memory).

    Stores file paths, seed IDs, and iteration positions so that the
    actual weight tensors can be loaded on demand via a loader callback.
    """

    __slots__ = ("file_paths", "seed_ids", "iter_positions", "n_features")

    def __init__(
        self,
        file_paths: list[Path],
        seed_ids: list[int],
        iter_positions: list[float],
        n_features: int,
    ) -> None:
        self.file_paths = file_paths
        self.seed_ids = seed_ids
        self.iter_positions = iter_positions
        self.n_features = n_features

    def __len__(self) -> int:
        return len(self.file_paths)


def load_weight_history_metadata(
    checkpoint_dir: Path,
    model_builder: Callable[[], torch.nn.Module],
    num_seeds: int | None = None,
) -> HistoryMetadata:
    """Collect history checkpoint metadata without loading weights.

    Scans ``checkpoint_dir`` for ``checkpoint_iter*.pt`` files in each
    seed directory and records their paths, seed IDs, and normalised
    iteration positions.  No weight tensors are materialised, so memory
    usage is negligible even for thousands of checkpoints.

    A single checkpoint is loaded once to determine ``n_features``
    (the flattened weight dimension).

    Args:
        checkpoint_dir: Root checkpoint directory.
        model_builder: Zero-argument callable that builds the target model.
        num_seeds: Maximum seed directories to scan (None = all).

    Returns:
        A ``HistoryMetadata`` instance.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    def _iter_from_name(path: Path) -> int:
        stem = path.stem
        if "checkpoint_iter" in stem:
            try:
                return int(stem.split("checkpoint_iter", maxsplit=1)[1])
            except Exception:
                return -1
        return 10**12

    seed_dirs = sorted(checkpoint_dir.glob("seed*"))
    if not seed_dirs:
        raise ValueError(f"No seed directories found in {checkpoint_dir}")

    logger.info(
        "Found %d seed directories in %s (metadata scan)", len(seed_dirs), checkpoint_dir,
    )

    file_paths: list[Path] = []
    seed_ids: list[int] = []
    iter_positions: list[float] = []

    loaded_seed_count = 0
    for seed_dir in seed_dirs:
        if num_seeds is not None and loaded_seed_count >= num_seeds:
            break

        seed_name = seed_dir.name
        if not seed_name.startswith("seed"):
            continue
        try:
            seed_id = int(seed_name[4:])
        except Exception:
            continue

        ckpt_files = sorted(seed_dir.glob("checkpoint_iter*.pt"), key=_iter_from_name)
        if not ckpt_files:
            continue
        if len(ckpt_files) == 1:
            logger.warning(
                "Only one checkpoint_iter found in %s; skipping history for this seed.",
                seed_dir,
            )
            continue
        ckpt_files = ckpt_files[:-1]  # drop last (≈ final.pt duplicate)

        n_ckpts = len(ckpt_files)
        for idx, ckpt_path in enumerate(ckpt_files):
            file_paths.append(ckpt_path)
            seed_ids.append(seed_id)
            if n_ckpts <= 1:
                iter_positions.append(1.0)
            else:
                iter_positions.append(float(idx) / float(n_ckpts - 1))

        loaded_seed_count += 1

    if not file_paths:
        raise ValueError(f"No valid checkpoint history found in {checkpoint_dir}")

    # Determine n_features from a single checkpoint
    sample_sd = torch.load(file_paths[0], map_location="cpu")
    if "model_state_dict" in sample_sd:
        sample_sd = sample_sd["model_state_dict"]
    model = model_builder()
    model.load_state_dict(sample_sd)
    wso = WeightSpaceObjectStateDict.from_model(model)
    n_features = wso.flatten(device="cpu").shape[0]
    del sample_sd, model, wso

    logger.info(
        "History metadata: %d checkpoints across %d seeds, n_features=%s",
        len(file_paths), loaded_seed_count, f"{n_features:,}",
    )
    return HistoryMetadata(file_paths, seed_ids, iter_positions, n_features)


def create_history_weight_loader(
    metadata: HistoryMetadata,
    model_builder: Callable[[], torch.nn.Module],
    device: str = "cpu",
) -> Callable[[int, int], torch.Tensor]:
    """Build a callback that loads a slice of history weights on demand.

    The returned callable has signature ``(start_idx, end_idx) -> Tensor``
    where the tensor has shape ``(end - start, n_features)`` and lives on
    ``device``.  Only the requested slice is loaded from disk, keeping
    peak memory proportional to the batch size rather than the full
    history.

    Internally, the model is built once to determine the parameter key
    order.  Subsequent loads extract and flatten parameters directly
    from the state-dict, avoiding repeated ``model_builder()`` and
    ``load_state_dict()`` calls (10-50x faster for large models).

    Args:
        metadata: ``HistoryMetadata`` from ``load_weight_history_metadata``.
        model_builder: Zero-argument callable that builds the target model.
        device: Device for the returned tensor.

    Returns:
        A weight-loader callback.
    """
    template = model_builder()
    param_keys = [name for name, _ in template.named_parameters()]
    del template

    def _load_batch(start_idx: int, end_idx: int) -> torch.Tensor:
        batch: list[torch.Tensor] = []
        for i in range(start_idx, end_idx):
            sd = torch.load(
                metadata.file_paths[i], map_location="cpu", weights_only=True,
            )
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            flat = torch.cat([sd[k].reshape(-1) for k in param_keys])
            batch.append(flat.to(device))
            del sd
        return torch.stack(batch)

    return _load_batch


def create_combined_weight_loader(
    final_weights: torch.Tensor,
    history_metadata: HistoryMetadata,
    model_builder: Callable[[], torch.nn.Module],
    device: str = "cpu",
) -> Callable[[int, int], torch.Tensor]:
    """Build a loader that serves final weights first, then history on demand.

    Indices ``[0, n_final)`` are served from the in-memory
    ``final_weights`` tensor.  Indices ``[n_final, n_final + n_history)``
    are loaded from disk via ``history_metadata``.

    Args:
        final_weights: Pre-loaded final checkpoint tensor (n_final, n_features).
        history_metadata: Metadata for history checkpoints.
        model_builder: Model builder for on-demand loading.
        device: Device for returned tensors.

    Returns:
        A weight-loader callback covering both final and history samples.
    """
    n_final = final_weights.size(0)
    history_loader = create_history_weight_loader(
        history_metadata, model_builder, device=device,
    )

    def _load_batch(start_idx: int, end_idx: int) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        # Portion from final weights (in-memory)
        if start_idx < n_final:
            final_end = min(end_idx, n_final)
            parts.append(final_weights[start_idx:final_end].to(device))

        # Portion from history (on-demand disk load)
        if end_idx > n_final:
            hist_start = max(start_idx, n_final) - n_final
            hist_end = end_idx - n_final
            parts.append(history_loader(hist_start, hist_end))

        return torch.cat(parts, dim=0)

    return _load_batch


def materialise_history_to_memmap(
    final_weights: torch.Tensor,
    history_metadata: "HistoryMetadata",
    model_builder: Callable[[], torch.nn.Module],
    batch_size: int = 200,
) -> tuple[np.memmap, Path]:
    """Load all weights (final + history) into a numpy memmap file.

    This is the expensive I/O step (one ``torch.load`` per checkpoint).
    The resulting memmap can be passed directly to
    :meth:`WeightSpacePCA.fit_streaming` for fast, GPU-only PCA.

    Args:
        final_weights: Pre-loaded final checkpoint tensor ``(n_final, D)``.
        history_metadata: Metadata from ``load_weight_history_metadata``.
        model_builder: Builds the target model (called once for key order).
        batch_size: Checkpoints loaded per logging tick.

    Returns:
        ``(memmap, path)`` – the memmap array of shape
        ``(n_final + n_history, D)`` and the path to the backing file
        (caller should ``Path.unlink()`` when done).
    """
    n_final = final_weights.shape[0]
    n_history = len(history_metadata)
    n_total = n_final + n_history
    n_features = history_metadata.n_features

    cache_path = Path(tempfile.mktemp(suffix=".pca_cache.mmap"))
    logger.info(
        "Materialising %d samples (%d final + %d history) to memmap "
        "(%.1f GB) at %s ...",
        n_total, n_final, n_history,
        n_total * n_features * 4 / 1e9,
        cache_path,
    )

    mmap = np.memmap(
        cache_path, dtype=np.float32, mode="w+",
        shape=(n_total, n_features),
    )

    # Final weights (already in memory)
    mmap[:n_final] = final_weights.cpu().numpy()

    # History weights (load from disk)
    template = model_builder()
    param_keys = [name for name, _ in template.named_parameters()]
    del template

    for start in range(0, n_history, batch_size):
        end = min(start + batch_size, n_history)
        for i in range(start, end):
            sd = torch.load(
                history_metadata.file_paths[i],
                map_location="cpu",
                weights_only=True,
            )
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            flat = torch.cat([sd[k].reshape(-1) for k in param_keys])
            mmap[n_final + i] = flat.numpy()
            del sd, flat
        if (start // batch_size) % 5 == 0:
            logger.info(
                "  memmap: %d / %d history checkpoints", end, n_history,
            )

    mmap.flush()
    logger.info("Materialisation complete.")
    return mmap, cache_path


def count_parameters_from_layout(layer_layout: list[int]) -> int:
    """Calculate total parameter count from layer layout.

    Args:
        layer_layout: List of layer sizes [input, hidden..., output].

    Returns:
        Total number of parameters (weights + biases).

    Example:
        >>> count_parameters_from_layout([784, 32, 32, 10])
        26506
    """
    total = 0
    for i in range(len(layer_layout) - 1):
        # Weight: out_features x in_features
        total += layer_layout[i] * layer_layout[i + 1]
        # Bias: out_features
        total += layer_layout[i + 1]
    return total
