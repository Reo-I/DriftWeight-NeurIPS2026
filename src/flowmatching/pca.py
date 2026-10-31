"""PCA dimensionality reduction for weight space.

This module provides PCA-based dimensionality reduction of neural network
weights using sklearn's IncrementalPCA. PCA is used to reduce the
dimensionality of the weight space before training Flow Matching,
which significantly improves training efficiency and generation quality.

Reference: DeepWeightFlow paper Section 2.3 and Appendix B.
https://arxiv.org/pdf/2601.05052

For MNIST MLP (784-32-32-10):
- Original dimension: 26,506 parameters
- PCA components: 99 (n_models - 1)
- This reduces the problem to learning in 99D space instead of 26,506D
"""

import gc
from collections.abc import Callable

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.utils.extmath import randomized_svd

from src.utils.logger import logger


class WeightSpacePCA:
    """PCA for weight space dimensionality reduction.

    Uses sklearn's IncrementalPCA to perform dimensionality reduction
    on flattened neural network weights. The PCA is fitted on the
    training weight distribution and used to transform weights to/from
    a lower-dimensional latent space.

    Attributes:
        n_components: Number of PCA components to keep.
        batch_size: Batch size for incremental fitting.
        fitted: Whether PCA has been fitted.
        pca: The underlying IncrementalPCA object.
        original_dim: Original dimension of weight space (set after fit).
    """

    def __init__(self, n_components: int, batch_size: int = 10) -> None:
        """Initialize WeightSpacePCA.

        Args:
            n_components: Number of components to keep.
            batch_size: Batch size for incremental PCA fitting.
        """
        self.n_components = n_components
        self.batch_size = batch_size
        self.fitted = False
        self.pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self.original_dim: int | None = None

    def _reset_pca_for_fit(self, n_samples: int, n_features: int) -> None:
        """Recreate IncrementalPCA with a safe batch size for current data.

        IncrementalPCA requires the first partial-fit batch to have at least
        ``n_components`` samples. If batch_size is smaller than n_components,
        sklearn can fail with shape inconsistency errors.
        """
        max_valid_components = min(n_samples - 1, n_features)
        if self.n_components > max_valid_components:
            raise ValueError(
                f"n_components={self.n_components} is invalid for "
                f"(n_samples={n_samples}, n_features={n_features}). "
                f"Maximum allowed is {max_valid_components}."
            )

        effective_batch_size = max(self.batch_size, self.n_components)
        effective_batch_size = min(effective_batch_size, n_samples)
        if effective_batch_size < self.n_components:
            raise ValueError(
                "IncrementalPCA requires batch_size >= n_components. "
                f"Got batch_size={effective_batch_size}, "
                f"n_components={self.n_components}."
            )

        if effective_batch_size != self.batch_size:
            logger.info(
                "Adjusting PCA batch_size from %d to %d to satisfy "
                "IncrementalPCA constraints (n_components=%d).",
                self.batch_size,
                effective_batch_size,
                self.n_components,
            )
        self.pca = IncrementalPCA(
            n_components=self.n_components, batch_size=effective_batch_size
        )

    def fit(self, weights: torch.Tensor | np.ndarray) -> "WeightSpacePCA":
        """Fit PCA on weight data.

        Args:
            weights: Weight data of shape (n_samples, n_features).
                     Can be torch.Tensor or numpy array.

        Returns:
            Self for method chaining.
        """
        # Convert to numpy if needed
        if isinstance(weights, torch.Tensor):
            weights_np = weights.cpu().numpy()
        else:
            weights_np = weights

        n_samples, n_features = weights_np.shape
        self.original_dim = n_features

        logger.info(f"Fitting PCA: {n_samples} samples, {n_features} features")
        logger.info(f"Target components: {self.n_components}")
        self._reset_pca_for_fit(n_samples=n_samples, n_features=n_features)

        # Fit PCA
        self.pca.fit(weights_np)
        self.fitted = True

        # Log explained variance
        total_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA fitted. Total variance captured: {total_variance:.4%}")
        logger.info(
            f"Dimension reduction: {n_features:,} -> {self.n_components} "
            f"({100 * self.n_components / n_features:.1f}%)"
        )

        return self

    @staticmethod
    def _to_device_float(batch: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        return batch.to(device, dtype=torch.float32)

    def fit_streaming(
        self,
        weight_loader: Callable[[int, int], torch.Tensor],
        n_total: int,
        n_features: int,
        batch_size: int = 200,
        device: str | None = None,
    ) -> "WeightSpacePCA":
        """Fit PCA via GPU-accelerated Gram-matrix decomposition.

        Reads data in ``batch_size``-sized slices via ``weight_loader``,
        keeping RAM bounded.  GPU (cuBLAS) is used for the heavy matrix
        products.  Only the upper triangle of the symmetric Gram matrix
        is computed, halving the number of loader calls in Pass 2.

        Args:
            weight_loader: ``(start_idx, end_idx) -> Tensor (n, n_features)``.
            n_total: Total number of samples.
            n_features: Dimensionality of each sample.
            batch_size: Samples loaded per batch.
            device: Compute device (``None`` → auto-detect CUDA).

        Returns:
            Self for method chaining.
        """
        import time as _time

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        self.original_dim = n_features
        n_components = min(self.n_components, n_total - 1)

        logger.info(
            "Fitting PCA (Gram, GPU): %d samples, %s features, "
            "batch_size=%d, n_components=%d, device=%s",
            n_total, f"{n_features:,}", batch_size, n_components, device,
        )
        t0 = _time.perf_counter()
        n_batches = (n_total + batch_size - 1) // batch_size

        def _load(s: int, e: int) -> torch.Tensor:
            b = weight_loader(s, e)
            return self._to_device_float(b, dev)

        # --- Pass 1: mean ---
        logger.info("Pass 1/3: computing mean (%d batches)...", n_batches)
        mean_acc = torch.zeros(n_features, dtype=torch.float64, device="cpu")
        for s in range(0, n_total, batch_size):
            e = min(s + batch_size, n_total)
            batch = _load(s, e)
            mean_acc += batch.cpu().to(torch.float64).sum(dim=0)
            del batch
        mean_cpu = (mean_acc / n_total).float()
        mean_dev = mean_cpu.to(dev)
        del mean_acc
        gc.collect()
        t1 = _time.perf_counter()
        logger.info("Pass 1/3: mean done (%.1fs).", t1 - t0)

        # --- Pass 2: Gram matrix (upper triangle) ---
        logger.info("Pass 2/3: Gram matrix (%d x %d)...", n_total, n_total)
        C = np.zeros((n_total, n_total), dtype=np.float32)

        for ib, i_s in enumerate(range(0, n_total, batch_size)):
            i_e = min(i_s + batch_size, n_total)
            bi = _load(i_s, i_e) - mean_dev.unsqueeze(0)

            for j_s in range(i_s, n_total, batch_size):
                j_e = min(j_s + batch_size, n_total)
                bj = bi if j_s == i_s else (_load(j_s, j_e) - mean_dev.unsqueeze(0))

                blk = (bi @ bj.T).cpu().numpy()
                C[i_s:i_e, j_s:j_e] = blk
                if j_s != i_s:
                    C[j_s:j_e, i_s:i_e] = blk.T
                    del bj

            del bi
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if (ib + 1) % 2 == 0 or ib + 1 == n_batches:
                logger.info(
                    "  Gram: row-block %d / %d (%.1fs)",
                    ib + 1, n_batches, _time.perf_counter() - t1,
                )

        del mean_dev
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        t2 = _time.perf_counter()
        logger.info("Pass 2/3: Gram done (%.1fs).", t2 - t1)

        # --- SVD on Gram matrix ---
        logger.info("SVD on Gram matrix (%d x %d)...", n_total, n_total)
        U, S, _ = randomized_svd(
            C, n_components=n_components, n_iter=5, random_state=42,
        )
        eigenvalues = S / (n_total - 1)
        del C
        gc.collect()

        # --- Pass 3: recover components ---
        logger.info("Pass 3/3: recovering components (%d batches)...", n_batches)
        components = torch.zeros(
            n_components, n_features, dtype=torch.float32, device=dev,
        )
        U_dev = torch.from_numpy(U).to(dev, dtype=torch.float32)
        mean_dev = mean_cpu.to(dev)

        for s in range(0, n_total, batch_size):
            e = min(s + batch_size, n_total)
            bc = _load(s, e) - mean_dev.unsqueeze(0)
            components += U_dev[s:e, :].T @ bc
            del bc

        if dev.type == "cuda":
            torch.cuda.empty_cache()

        norms = torch.norm(components, dim=1, keepdim=True)
        components = components / torch.clamp(norms, min=1e-10)
        components_np = components.cpu().numpy()

        del components, U_dev, mean_dev
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        t3 = _time.perf_counter()
        logger.info("Pass 3/3: components done (%.1fs).", t3 - t2)

        # --- Store results ---
        total_var = float(np.sum(eigenvalues))
        ev_ratio = (
            eigenvalues[:n_components] / total_var
            if total_var > 0
            else eigenvalues[:n_components]
        )

        self.pca = IncrementalPCA(n_components=n_components)
        self.pca.components_ = components_np
        self.pca.mean_ = mean_cpu.numpy()
        self.pca.var_ = np.ones(n_features, dtype=np.float32)
        self.pca.explained_variance_ = eigenvalues[:n_components]
        self.pca.explained_variance_ratio_ = ev_ratio
        self.pca.n_components_ = n_components
        self.n_components = n_components
        self.fitted = True

        captured = float(np.sum(ev_ratio))
        logger.info(
            "PCA fitted. Variance captured: %.4f%% | Total: %.1fs",
            captured * 100, t3 - t0,
        )
        logger.info(
            "Dimension reduction: %s -> %d (%.1f%%)",
            f"{n_features:,}", n_components, 100 * n_components / n_features,
        )

        del mean_cpu, components_np
        gc.collect()
        return self

    def transform_streaming(
        self,
        weight_loader: Callable[[int, int], torch.Tensor],
        n_total: int,
        batch_size: int = 200,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Transform data to PCA space in a streaming fashion.

        Loads batches via ``weight_loader``, transforms each, and
        concatenates the results into a single tensor in PCA space.
        Peak memory is O(batch_size * n_features) instead of
        O(n_total * n_features).

        Args:
            weight_loader: Callable(start_idx, end_idx) -> Tensor of shape
                (end-start, n_features).
            n_total: Total number of samples.
            batch_size: Samples per transform batch.
            device: Device for the output tensor.

        Returns:
            Transformed tensor of shape (n_total, n_components).
        """
        if not self.fitted:
            raise RuntimeError("PCA must be fitted before transform_streaming")

        chunks: list[torch.Tensor] = []
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch = weight_loader(start, end)
            transformed = self.transform(batch)
            chunks.append(transformed.to(device))
            del batch
            if (start // batch_size) % 5 == 0:
                logger.info("PCA transform: %d / %d samples", end, n_total)

        return torch.cat(chunks, dim=0)

    def transform(self, weights: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Transform weights to PCA space.

        Args:
            weights: Weight data of shape (n_samples, n_features).

        Returns:
            Transformed data of shape (n_samples, n_components) as torch.Tensor.

        Raises:
            RuntimeError: If PCA has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("PCA must be fitted before transform")

        # Convert to numpy if needed
        if isinstance(weights, torch.Tensor):
            weights_np = weights.cpu().numpy()
            device = weights.device
        else:
            weights_np = weights
            device = "cpu"

        # Transform
        transformed = self.pca.transform(weights_np)

        return torch.tensor(transformed, dtype=torch.float32, device=device)

    def inverse_transform(
        self, latent: torch.Tensor | np.ndarray, device: str = "cpu"
    ) -> torch.Tensor:
        """Transform from PCA space back to weight space.

        Args:
            latent: Latent codes of shape (n_samples, n_components).
            device: Device for the output tensor.

        Returns:
            Reconstructed weights of shape (n_samples, n_features) as torch.Tensor.

        Raises:
            RuntimeError: If PCA has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("PCA must be fitted before inverse_transform")

        # Convert to numpy if needed
        if isinstance(latent, torch.Tensor):
            latent_np = latent.cpu().numpy()
        else:
            latent_np = latent

        # Inverse transform
        reconstructed = self.pca.inverse_transform(latent_np)

        return torch.tensor(reconstructed, dtype=torch.float32, device=device)

    def fit_transform(self, weights: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Fit PCA and transform weights in one step.

        Args:
            weights: Weight data of shape (n_samples, n_features).

        Returns:
            Transformed data of shape (n_samples, n_components) as torch.Tensor.
        """
        # Convert to numpy if needed
        if isinstance(weights, torch.Tensor):
            weights_np = weights.cpu().numpy()
            device = weights.device
        else:
            weights_np = weights
            device = "cpu"

        n_samples, n_features = weights_np.shape
        self.original_dim = n_features

        logger.info(f"Fitting PCA: {n_samples} samples, {n_features} features")
        logger.info(f"Target components: {self.n_components}")
        self._reset_pca_for_fit(n_samples=n_samples, n_features=n_features)

        # Fit and transform
        transformed = self.pca.fit_transform(weights_np)
        self.fitted = True

        # Log explained variance
        total_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA fitted. Total variance captured: {total_variance:.4%}")
        logger.info(
            f"Dimension reduction: {n_features:,} -> {self.n_components} "
            f"({100 * self.n_components / n_features:.1f}%)"
        )

        return torch.tensor(transformed, dtype=torch.float32, device=device)

    @property
    def explained_variance_ratio(self) -> np.ndarray | None:
        """Get explained variance ratio for each component."""
        if not self.fitted:
            return None
        return self.pca.explained_variance_ratio_

    @property
    def total_variance_captured(self) -> float:
        """Get total variance captured by PCA components."""
        if not self.fitted:
            return 0.0
        return float(np.sum(self.pca.explained_variance_ratio_))

    def save(self, path: str) -> None:
        """Save PCA parameters to file.

        Args:
            path: Path to save the PCA parameters.
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted PCA")

        np.savez(
            path,
            components=self.pca.components_,
            mean=self.pca.mean_,
            var=self.pca.var_,
            explained_variance=self.pca.explained_variance_,
            explained_variance_ratio=self.pca.explained_variance_ratio_,
            n_components=self.n_components,
            original_dim=self.original_dim,
        )
        logger.info(f"PCA saved to {path}")

    @classmethod
    def load(cls, path: str) -> "WeightSpacePCA":
        """Load PCA parameters from file.

        Args:
            path: Path to load the PCA parameters from.

        Returns:
            Loaded WeightSpacePCA instance.
        """
        data = np.load(path)

        n_components = int(data["n_components"])
        pca = cls(n_components=n_components)

        # Restore PCA state
        pca.pca.components_ = data["components"]
        pca.pca.mean_ = data["mean"]
        pca.pca.var_ = data["var"]
        pca.pca.explained_variance_ = data["explained_variance"]
        pca.pca.explained_variance_ratio_ = data["explained_variance_ratio"]
        pca.pca.n_components_ = n_components
        pca.original_dim = int(data["original_dim"])
        pca.fitted = True

        logger.info(f"PCA loaded from {path}")
        logger.info(
            f"Components: {n_components}, Original dim: {pca.original_dim}, "
            f"Variance captured: {pca.total_variance_captured:.4%}"
        )

        return pca


def apply_pca_to_weights(
    weights: torch.Tensor,
    n_components: int | None = None,
    return_pca: bool = True,
) -> tuple[torch.Tensor, WeightSpacePCA | None]:
    """Apply PCA reduction to weight data.

    Convenience function that creates a WeightSpacePCA, fits it, and
    transforms the weights in one call.

    Args:
        weights: Weight data of shape (n_samples, n_features).
        n_components: Number of components. If None, uses min(n_samples-1, n_features).
        return_pca: Whether to return the fitted PCA object.

    Returns:
        Tuple of (reduced_weights, pca_object).
        If return_pca is False, pca_object will be None.

    Example:
        >>> weights = torch.randn(100, 26506)  # 100 models, 26506 params
        >>> reduced, pca = apply_pca_to_weights(weights, n_components=99)
        >>> reduced.shape
        torch.Size([100, 99])
    """
    n_samples, n_features = weights.shape

    # Default: use n_samples - 1 components (max possible for n samples)
    if n_components is None:
        n_components = min(n_samples - 1, n_features)

    logger.info(f"Applying PCA: {n_features} -> {n_components} dimensions")

    pca = WeightSpacePCA(n_components=n_components)
    reduced_weights = pca.fit_transform(weights)

    if return_pca:
        return reduced_weights, pca
    else:
        return reduced_weights, None
