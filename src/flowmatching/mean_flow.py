"""MeanFlow training and generation for weight space.

This module implements the MeanFlow algorithm for learning to generate
neural network weights. MeanFlow uses average velocity instead of
instantaneous velocity, enabling efficient one/few-step generation.

The core idea is the MeanFlow Identity:
    u(z_t, r, t) = v(z_t, t) - (t - r) * du/dt

where u is the average velocity and v is the instantaneous velocity.

Reference:
    "Mean Flows for One-step Generative Modeling"
    (https://arxiv.org/pdf/2505.13447)
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.flowmatching.drifting import drifting_loss_rawspace
from src.utils.logger import logger


def _stopgrad(x: torch.Tensor) -> torch.Tensor:
    """Stop gradient propagation (detach tensor).

    Args:
        x: Input tensor.

    Returns:
        Detached tensor.
    """
    return x.detach()


def adaptive_l2_loss(
    error: torch.Tensor,
    gamma: float = 0.5,
    c: float = 1e-3,
) -> torch.Tensor:
    """Adaptive L2 loss from the MeanFlow paper.

    Computes sg(w) * ||delta||_2^2, where w = 1 / (||delta||^2 + c)^p
    and p = 1 - gamma. This is equivalent to minimizing ||delta||^{2*gamma}.

    Args:
        error: Residual tensor of shape (batch_size, dim).
        gamma: Power parameter. gamma=0.5 is Pseudo-Huber loss,
               gamma=1.0 is the paper's best setting for ImageNet.
        c: Small constant for numerical stability.

    Returns:
        Scalar loss value.
    """
    # Per-sample squared L2 norm
    delta_sq = torch.mean(error ** 2, dim=-1)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (_stopgrad(w) * delta_sq).mean()


class WeightSpaceMeanFlow:
    """MeanFlow trainer for weight space generation.

    Implements the MeanFlow algorithm for learning to transport samples
    from a source distribution (Gaussian noise) to a target distribution
    (trained model weights) using average velocity fields.

    Key differences from Flow Matching:
        - Conditions on two time variables (t, r) instead of one
        - Uses JVP (Jacobian-Vector Product) for computing du/dt
        - Adaptive L2 loss instead of simple MSE
        - Supports efficient few-step or one-step generation

    Attributes:
        device: Device for computation.
        flow_ratio: Ratio of samples where r=t (instantaneous velocity).
        t1_ratio: Ratio of samples forced to t=1.
        r0_ratio: Ratio of samples forced to r=0.
        onestep_lambda: Weight for 1-step auxiliary loss.
        time_dist: Time distribution type ('uniform' or 'lognorm').
        time_dist_mu: Mean for lognorm distribution.
        time_dist_sigma: Std for lognorm distribution.
        loss_gamma: Gamma for adaptive L2 loss.
        sigma: Noise scale added to interpolated points.
        best_loss: Best training loss achieved.
        best_model_state: State dict of model at best loss.
        input_dim: Dimension of weight space (set during training).

    Example:
        >>> mf = WeightSpaceMeanFlow(
        ...     targetloader=weight_loader,
        ...     model=meanflow_model,
        ...     source_std=0.01,
        ...     device='cuda',
        ... )
        >>> mf.train(n_iters=30000, optimizer=optimizer)
        >>> generated = mf.generate_weights(n_samples=100)
    """

    def __init__(
        self,
        targetloader: DataLoader,
        model: nn.Module,
        history_by_seed: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        device: str | torch.device | None = None,
        source_std: float = 0.01,
        sigma: float = 0.001,
        flow_ratio: float = 0.75,
        t1_ratio: float = 0.0,
        r0_ratio: float = 0.0,
        onestep_lambda: float = 1.0,
        time_dist: str = "uniform",
        time_dist_mu: float = -0.4,
        time_dist_sigma: float = 1.0,
        loss_gamma: float = 0.0,
        jvp_api: str = "autograd",
        use_drift_loss: bool = False,
        drift_lambda: float = 1.0,
        drift_target_ratio: float = 0.1,
        drift_auto_scale: str = "ema",
        drift_temperatures: list[float] | None = None,
        drift_neg_max_total: int = 32,
        drift_same_ratio: float = 0.2,
        drift_history_tail_exclude: float = 0.05,
        drift_use_a6_norm: bool = True,
        drift_a6_eps: float = 1e-6,
        drift_hardneg_k: int = 64,
        drift_ema_decay: float = 0.99,
        drift_scale_eps: float = 1e-8,
    ) -> None:
        """Initialize WeightSpaceMeanFlow.

        Unlike FlowMatching, MeanFlow does NOT use a source DataLoader.
        Instead, fresh noise is generated inside compute_loss() every
        iteration (matching the reference MeanFlow implementation).

        Args:
            targetloader: DataLoader for target distribution (trained weights).
            model: Neural network model for predicting average velocity
                   u(z_t, r, t). Must accept (x, t, r) as inputs.
            device: Device for computation.
            source_std: Standard deviation for source noise distribution.
                Fresh noise is sampled as N(0, source_std^2) each iteration.
            sigma: Noise scale added to interpolated points for stability.
            flow_ratio: Fraction of batch where r=t (FM training signal).
                Paper best: 0.75 (25% r!=t, Table 1a).
            t1_ratio: Fraction of batch where t is forced to 1.
            r0_ratio: Fraction of batch where r is forced to 0.
            onestep_lambda: Weight of 1-step auxiliary loss.
            time_dist: Distribution for sampling (t, r) pairs.
            time_dist_mu: Mean parameter for lognorm distribution.
            time_dist_sigma: Std parameter for lognorm distribution.
            loss_gamma: Gamma parameter for adaptive L2 loss (p=1-gamma).
                Paper best: 0.0 (p=1.0, Table 1e). 0.5 = Pseudo-Huber.
            jvp_api: JVP computation backend.
                'autograd' uses torch.autograd.functional.jvp,
                'functorch' uses torch.func.jvp.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.targetloader = targetloader
        self.model = model.to(self.device)
        self.source_std = source_std
        self.sigma = sigma

        # MeanFlow-specific settings
        self.flow_ratio = min(max(float(flow_ratio), 0.0), 1.0)
        self.t1_ratio = min(max(float(t1_ratio), 0.0), 1.0)
        self.r0_ratio = min(max(float(r0_ratio), 0.0), 1.0)
        if self.flow_ratio + self.t1_ratio + self.r0_ratio > 1.0:
            raise ValueError(
                "flow_ratio + t1_ratio + r0_ratio must be <= 1.0. "
                f"Got flow_ratio={self.flow_ratio}, t1_ratio={self.t1_ratio}, "
                f"r0_ratio={self.r0_ratio}."
            )
        self.onestep_lambda = max(float(onestep_lambda), 0.0)
        self.time_dist = time_dist
        self.time_dist_mu = time_dist_mu
        self.time_dist_sigma = time_dist_sigma
        self.loss_gamma = loss_gamma
        self.use_drift_loss = use_drift_loss
        self.drift_lambda = drift_lambda
        self.drift_target_ratio = drift_target_ratio
        self.drift_auto_scale = drift_auto_scale
        self.drift_temperatures = drift_temperatures or [0.02, 0.05, 0.2]
        self.drift_neg_max_total = drift_neg_max_total
        self.drift_same_ratio = min(max(float(drift_same_ratio), 0.0), 1.0)
        self.drift_history_tail_exclude = min(
            max(float(drift_history_tail_exclude), 0.0), 0.999
        )
        self.drift_use_a6_norm = bool(drift_use_a6_norm)
        self.drift_a6_eps = max(float(drift_a6_eps), 1e-12)
        self.drift_hardneg_k = max(int(drift_hardneg_k), 0)
        self.drift_ema_decay = drift_ema_decay
        self.drift_scale_eps = drift_scale_eps
        self._ema_base: float | None = None
        self._ema_drift: float | None = None
        self._last_lambda_eff: float = 0.0
        self._last_grad_base: float = 0.0
        self._last_grad_drift: float = 0.0
        self._last_same_count: int = 0
        self._last_other_count: int = 0
        self._last_sampling_summary: str = ""
        self._history_by_seed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._target_history_bank: torch.Tensor | None = None
        self._bank_seed_ids = torch.empty(0, device=self.device, dtype=torch.long)
        self._bank_iter_pos = torch.empty(0, dtype=torch.float32)

        # JVP backend selection
        if jvp_api == "functorch":
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        else:
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

        # Tracking
        self.best_loss = float("inf")
        self.best_model_state: dict | None = None
        self.input_dim: int | None = None

        # Dataloader iterator (target only; noise is generated fresh)
        self._target_iter: iter | None = None
        if self.use_drift_loss:
            if history_by_seed is None:
                raise ValueError(
                    "Drift loss requires explicit seed-indexed history input: "
                    "history_by_seed."
                )
            self._set_history_bank(history_by_seed=history_by_seed)

    def _set_history_bank(
            self,
            history_by_seed: dict[int, tuple[torch.Tensor, torch.Tensor]],
        ) -> None:
        """Build seed-indexed history bank for drift negative sampling."""
        if not history_by_seed:
            raise ValueError("history_by_seed must not be empty.")

        keep_limit = 1.0 - self.drift_history_tail_exclude
        filtered_by_seed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        bank_chunks: list[torch.Tensor] = []
        seed_chunks: list[torch.Tensor] = []
        pos_chunks: list[torch.Tensor] = []

        for seed_val in sorted(history_by_seed.keys()):
            seed_x, seed_pos = history_by_seed[seed_val]
            if seed_x.ndim != 2 or seed_pos.ndim != 1:
                raise ValueError(
                    "Each history_by_seed value must be (x:[N,D], iter_pos:[N])."
                )
            if seed_x.size(0) != seed_pos.size(0):
                raise ValueError(
                    f"history_by_seed[{seed_val}] has mismatched lengths: "
                    f"x={seed_x.size(0)}, iter_pos={seed_pos.size(0)}"
                )
            if seed_x.size(0) == 0:
                continue

            seed_x = seed_x.detach().to(self.device)
            seed_pos = seed_pos.detach().to(self.device).float()
            keep_mask = seed_pos < keep_limit
            if not keep_mask.any():
                continue
            seed_x = seed_x[keep_mask]
            seed_pos = seed_pos[keep_mask]
            order = torch.argsort(seed_pos)
            seed_x = seed_x[order]
            seed_pos = seed_pos[order]

            filtered_by_seed[int(seed_val)] = (seed_x, seed_pos)
            bank_chunks.append(seed_x)
            seed_chunks.append(
                torch.full(
                    (seed_x.size(0),),
                    int(seed_val),
                    device=self.device,
                    dtype=torch.long,
                )
            )
            pos_chunks.append(seed_pos)

        if not bank_chunks:
            raise ValueError("History bank is empty after filtering.")
        if not filtered_by_seed:
            raise ValueError("No valid seed groups remain in history bank.")

        self._history_by_seed = filtered_by_seed
        self._target_history_bank = torch.cat(bank_chunks, dim=0)
        self._bank_seed_ids = torch.cat(seed_chunks, dim=0)
        self._bank_iter_pos = torch.cat(pos_chunks, dim=0)

        logger.info(
            "Prepared MeanFlow seed history index: %d seeds, %d checkpoints",
            len(self._history_by_seed),
            int(self._target_history_bank.shape[0]),
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return parameters optimized during training."""
        return list(self.model.parameters())

    def _compute_grad_norm(
        self,
        loss: torch.Tensor,
        parameters: list[nn.Parameter],
    ) -> float:
        grads = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        total = 0.0
        for grad in grads:
            if grad is not None:
                total += float((grad.detach().float() ** 2).sum().item())
        return float(np.sqrt(total))

    def _compute_lambda_eff(
            self,
            base_loss: torch.Tensor,
            drift_loss: torch.Tensor,
            parameters: list[nn.Parameter],
        ) -> float:
        if self.drift_auto_scale == "none":
            self._last_grad_base = 0.0
            self._last_grad_drift = 0.0
            return self.drift_lambda

        if self.drift_auto_scale == "grad":
            g_base = self._compute_grad_norm(base_loss, parameters)
            g_drift = self._compute_grad_norm(drift_loss, parameters)
            self._last_grad_base = g_base
            self._last_grad_drift = g_drift
            return (
                self.drift_lambda
                * (g_base / (g_drift + self.drift_scale_eps))
                * self.drift_target_ratio
            )

        # Default EMA scaling
        base_val = float(base_loss.detach().item())
        drift_val = float(drift_loss.detach().item())
        if self._ema_base is None:
            self._ema_base = base_val
            self._ema_drift = drift_val
        else:
            decay = self.drift_ema_decay
            self._ema_base = decay * self._ema_base + (1.0 - decay) * base_val
            self._ema_drift = decay * self._ema_drift + (1.0 - decay) * drift_val
        self._last_grad_base = 0.0
        self._last_grad_drift = 0.0
        assert self._ema_base is not None and self._ema_drift is not None
        return (
            self.drift_lambda
            * (self._ema_base / (self._ema_drift + self.drift_scale_eps))
            * self.drift_target_ratio
        )

    @staticmethod
    def _split_quota(total: int, bins: int) -> list[int]:
        """Split total count across bins as evenly as possible."""
        if bins <= 0:
            return []
        base = total // bins
        rem = total % bins
        return [base + (1 if i < rem else 0) for i in range(bins)]

    def _sample_history_indices(
            self,
            weight_matrix: torch.Tensor,
            count: int,
            quota: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample history vectors for each batch row from row-wise weights."""
        if count <= 0:
            dim = self._target_history_bank.size(1) if self._target_history_bank is not None else 0
            return (
                torch.empty((0, dim), device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
            )
        if self._target_history_bank is None:
            raise RuntimeError("History bank is not initialized.")

        row_sums = weight_matrix.sum(dim=1)
        needs = quota > 0
        if torch.any(needs & (row_sums <= 0)):
            raise RuntimeError(
                "Requested drift negatives but one or more rows had empty candidate pools."
            )

        safe_weights = weight_matrix.clone()
        zero_rows = row_sums <= 0
        if zero_rows.any():
            safe_weights[zero_rows, 0] = 1.0

        pick_idx = torch.multinomial(safe_weights, num_samples=count, replacement=True)
        picked = self._target_history_bank[pick_idx]  # [B, count, D]
        take_mask = (
            torch.arange(count, device=self.device).unsqueeze(0)
            < quota.unsqueeze(1)
        )
        return picked[take_mask], pick_idx[take_mask]

    @staticmethod
    def _iter_stats_str(pos: torch.Tensor) -> str:
        """Format iteration-position summary for debug logging."""
        if pos.numel() == 0:
            return "n=0"
        q = torch.quantile(pos.float(), torch.tensor([0.1, 0.5, 0.9]))
        return (
            f"n={int(pos.numel())}, mean={float(pos.mean()):.3f}, "
            f"p10/p50/p90={float(q[0]):.3f}/{float(q[1]):.3f}/{float(q[2]):.3f}"
        )

    @staticmethod
    def _seed_count_str(seed_tensor: torch.Tensor, topk: int = 8) -> str:
        """Format top-k seed histogram for debug logging."""
        if seed_tensor.numel() == 0:
            return "none"
        uniq, cnt = torch.unique(seed_tensor.long(), return_counts=True)
        order = torch.argsort(cnt, descending=True)
        items: list[str] = []
        for idx in order[:topk]:
            items.append(f"{int(uniq[idx])}:{int(cnt[idx])}")
        return ", ".join(items)

    def _format_sampling_summary(
        self,
        seed_ids_dev: torch.Tensor,
        same_idx: torch.Tensor,
        other_idx: torch.Tensor,
        progress: float,
    ) -> str:
        """Build sampling summary string while preserving current log format."""
        target_seed_vals = seed_ids_dev.detach().cpu().tolist()
        same_seed_vals = self._bank_seed_ids[same_idx].detach().cpu()
        other_seed_vals = self._bank_seed_ids[other_idx].detach().cpu()
        same_pos_vals = self._bank_iter_pos[same_idx].detach().cpu()
        other_pos_vals = self._bank_iter_pos[other_idx].detach().cpu()
        return (
            f"target_batch_seeds={target_seed_vals} | "
            f"same_seed_counts={self._seed_count_str(same_seed_vals)} | "
            f"other_seed_counts={self._seed_count_str(other_seed_vals)} | "
            f"same_iter_stats=({self._iter_stats_str(same_pos_vals)}) | "
            f"other_iter_stats=({self._iter_stats_str(other_pos_vals)}) | "
            f"progress={float(progress):.3f}"
        )

    def _truncate_negatives_keep_head(
        self,
        y_neg: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        """Keep current negatives at head while truncating overflow."""
        if y_neg.size(0) <= self.drift_neg_max_total:
            return y_neg
        keep_head = current.size(0)
        overflow = self.drift_neg_max_total - keep_head
        if overflow > 0:
            tail = y_neg[keep_head:]
            idx = torch.randperm(tail.size(0), device=tail.device)[:overflow]
            return torch.cat([current, tail[idx]], dim=0)
        return current


    def _sample_drift_negatives(
            self,
            x_gen: torch.Tensor,
            seed_ids: torch.Tensor,
            progress: float,
            global_step: int = -1,
        ) -> torch.Tensor:
        """Compose y_neg from three sources.

        1) current generated samples (always included at head),
        2) same-seed history (tail-excluded),
        3) different-seed history (tail-excluded).
        """
        current = x_gen.detach()
        batch_size = current.size(0)
        remaining = max(0, self.drift_neg_max_total - batch_size)
        n_same_total = int(remaining * self.drift_same_ratio)
        n_other_total = remaining - n_same_total

        same_chunks: list[torch.Tensor] = []
        other_chunks: list[torch.Tensor] = []

        has_history = self._target_history_bank is not None and bool(self._history_by_seed)
        if not has_history:
            raise RuntimeError(
                "Drift negative sampling requires a non-empty history bank."
            )
        same_idx = torch.empty((0,), dtype=torch.long, device=self.device)
        other_idx = torch.empty((0,), dtype=torch.long, device=self.device)
        if batch_size > 0 and has_history:
            assert self._target_history_bank is not None
            seed_ids_dev = seed_ids.to(device=self.device, dtype=torch.long)
            same_quota = torch.tensor(
                self._split_quota(n_same_total, batch_size),
                device=self.device,
                dtype=torch.long,
            )
            other_quota = torch.tensor(
                self._split_quota(n_other_total, batch_size),
                device=self.device,
                dtype=torch.long,
            )

            progress_t = torch.tensor(float(progress), device=self.device, dtype=torch.float32)
            sigma = 0.2
            base_weights = torch.exp(
                -0.5 * ((self._bank_iter_pos.float() - progress_t) / sigma) ** 2
            )
            base_weights = torch.clamp(base_weights, min=1e-12)

            same_mask = self._bank_seed_ids.unsqueeze(0) == seed_ids_dev.unsqueeze(1)
            other_mask = ~same_mask
            same_w = base_weights.unsqueeze(0) * same_mask.float()
            other_w = base_weights.unsqueeze(0) * other_mask.float()

            same_count = int(same_quota.max().item()) if same_quota.numel() > 0 else 0
            other_count = int(other_quota.max().item()) if other_quota.numel() > 0 else 0

            if same_count > 0:
                same, same_idx = self._sample_history_indices(same_w, same_count, same_quota)
                if same.numel() > 0:
                    same_chunks.append(same)
            else:
                same_idx = torch.empty((0,), dtype=torch.long, device=self.device)
            if other_count > 0:
                other, other_idx = self._sample_history_indices(other_w, other_count, other_quota)
                if other.numel() > 0:
                    other_chunks.append(other)
            self._last_sampling_summary = self._format_sampling_summary(
                seed_ids_dev=seed_ids_dev,
                same_idx=same_idx,
                other_idx=other_idx,
                progress=progress,
            )

        same = (
            torch.cat(same_chunks, dim=0)
            if same_chunks
            else current.new_empty((0, current.size(1)))
        )
        other = (
            torch.cat(other_chunks, dim=0)
            if other_chunks
            else current.new_empty((0, current.size(1)))
        )
        if same.device != self.device or same.dtype != current.dtype:
            same = same.to(device=self.device, dtype=current.dtype)
        if other.device != self.device or other.dtype != current.dtype:
            other = other.to(device=self.device, dtype=current.dtype)

        y_neg = torch.cat([current, same, other], dim=0)
        self._last_same_count = int(same.size(0))
        self._last_other_count = int(other.size(0))
        return self._truncate_negatives_keep_head(y_neg=y_neg, current=current)


    def _sample_from_loader(
            self,
            loader: DataLoader,
            is_source: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch from the target dataloader with automatic restart.

        Args:
            loader: DataLoader to sample from.
            is_source: Unused (kept for API compatibility). MeanFlow
                generates fresh noise internally, not from a DataLoader.

        Returns:
            Batch tensor on the correct device.
        """
        try:
            if self._target_iter is None:
                self._target_iter = iter(loader)
            batch = next(self._target_iter)
        except StopIteration:
            self._target_iter = iter(loader)
            batch = next(self._target_iter)

        if isinstance(batch, (tuple, list)):
            if len(batch) >= 3:
                return (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                    batch[2].to(self.device),
                )
            if len(batch) >= 2:
                return batch[0].to(self.device), batch[1].to(self.device)
            return batch[0].to(self.device)

        return batch.to(self.device)

    def sample_t_r(
            self,
            batch_size: int,
            device: str | torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (t, r) time pairs for MeanFlow training.

        Samples two time values, assigns the larger to t and smaller to r.
        A fraction (flow_ratio) of pairs have r=t, which reduces to standard
        Flow Matching (instantaneous velocity learning).

        Args:
            batch_size: Number of (t, r) pairs to sample.
            device: Device for output tensors.

        Returns:
            Tuple of (t, r) tensors each of shape (batch_size,).
        """
        if self.time_dist == "uniform":
            samples = torch.rand(batch_size, 2, device=device)
        elif self.time_dist == "lognorm":
            mu, sigma = self.time_dist_mu, self.time_dist_sigma
            normal_samples = torch.randn(batch_size, 2, device=device) * sigma + mu
            # Apply sigmoid to map to [0, 1]
            samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time distribution: {self.time_dist}")

        # Assign t = max, r = min for each pair
        t = torch.maximum(samples[:, 0], samples[:, 1])
        r = torch.minimum(samples[:, 0], samples[:, 1])

        # Mix in explicit supervision subsets with disjoint indices:
        # 1) r=t (Flow Matching-like signal)
        # 2) t=1
        # 3) r=0
        num_rt = int(self.flow_ratio * batch_size)
        num_t1 = int(self.t1_ratio * batch_size)
        num_r0 = int(self.r0_ratio * batch_size)
        perm = torch.randperm(batch_size, device=device)
        rt_idx = perm[:num_rt]
        t1_idx = perm[num_rt : num_rt + num_t1]
        r0_idx = perm[num_rt + num_t1 : num_rt + num_t1 + num_r0]
        r[rt_idx] = t[rt_idx]
        t[t1_idx] = 1.0
        r[r0_idx] = 0.0
        return t, r

    def _one_step_generate(self, z1: torch.Tensor) -> torch.Tensor:
        """One-step map from noise (t=1) to data side (r=0)."""
        batch_size = z1.size(0)
        t_one = torch.ones((batch_size, 1), device=z1.device, dtype=z1.dtype)
        r_zero = torch.zeros((batch_size, 1), device=z1.device, dtype=z1.dtype)
        u_10 = self.model(z1, t_one, r_zero)
        return z1 - (t_one - r_zero) * u_10

    def _two_step_generate(self, z1: torch.Tensor) -> torch.Tensor:
        """Two-step map (1 -> 1-d -> 0) used as a stop-grad teacher.

        d is sampled uniformly from (0, 1) per batch element and interpreted
        as the first-step travel distance from t=1.
        """
        batch_size = z1.size(0)
        t_one = torch.ones((batch_size, 1), device=z1.device, dtype=z1.dtype)
        t_zero = torch.zeros((batch_size, 1), device=z1.device, dtype=z1.dtype)
        d = torch.rand((batch_size, 1), device=z1.device, dtype=z1.dtype).clamp(
            0.05, 0.95
        )
        t_mid = t_one - d
        u_1m = self.model(z1, t_one, t_mid)
        z_mid = z1 - (t_one - t_mid) * u_1m
        u_m0 = self.model(z_mid, t_mid, t_zero)
        return z_mid - (t_mid - t_zero) * u_m0

    def _compute_base_terms(
        self,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MeanFlow base loss terms (without auxiliary/drift)."""
        batch_size = x1.shape[0]
        device = x1.device

        t, r = self.sample_t_r(batch_size, device)

        # Expand t and r for broadcasting: (batch_size,) -> (batch_size, 1)
        t_expanded = t.unsqueeze(-1).detach().clone()
        r_expanded = r.unsqueeze(-1).detach().clone()

        # Generate FRESH noise every iteration (critical for MeanFlow JVP)
        # Reference: MeanFlow/meanflow.py line 136: e = torch.randn_like(x)
        e = torch.randn_like(x1) * self.source_std

        # Linear interpolation: z_t = (1 - t) * x1 + t * e
        # MeanFlow convention: t=0 is data, t=1 is noise
        z = (1 - t_expanded) * x1 + t_expanded * e

        # Add small noise for stability
        epsilon = torch.randn_like(z) * self.sigma
        z = z + epsilon

        # Instantaneous velocity: v = e - x1 (noise - data)
        v = e - x1

        # Compute u and du/dt via JVP
        # The JVP tangent vector is (v, 1, 0) for (dz/dt, dt/dt, dr/dt)
        model_fn = lambda z_in, t_in, r_in: self.model(z_in, t_in, r_in)

        jvp_args = (
            model_fn,
            (z, t_expanded, r_expanded),
            (v, torch.ones_like(t_expanded), torch.zeros_like(r_expanded)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        # MeanFlow target: u_tgt = v - (t - r) * du/dt
        u_tgt = v - (t_expanded - r_expanded) * dudt

        # Compute base loss
        error = u - _stopgrad(u_tgt)
        base_loss = adaptive_l2_loss(error, gamma=self.loss_gamma)

        # MSE for logging
        mse_val = (_stopgrad(error) ** 2).mean()
        return base_loss, mse_val

    def _compute_onestep_aux_consistency(
            self,
            x1: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one-step consistency auxiliary loss."""
        z1 = torch.randn_like(x1) * self.source_std
        x_hat_10 = self._one_step_generate(z1)
        with torch.no_grad():
            x_two_step = self._two_step_generate(z1)
        onestep_aux_loss = F.mse_loss(x_hat_10, x_two_step)
        return x_hat_10, onestep_aux_loss


    def _update_metrics_with_drift(
        self,
        metrics: dict[str, float],
        total_loss: torch.Tensor,
        drift_out,
        lambda_eff: float,
        seed_ids: torch.Tensor,
        t_xgen_ms: float,
        t_neg_ms: float,
    ) -> None:
        """Apply drift-specific metric values in place."""
        t_compute_v_ms = float(drift_out.compute_v_seconds * 1000.0)
        metrics.update(
            {
                "drift_loss": float(drift_out.loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "mean_abs_v": float(drift_out.mean_abs_v),
                "lambda_drift_eff": float(lambda_eff),
                "ema_base": float(self._ema_base) if self._ema_base is not None else 0.0,
                "ema_drift": float(self._ema_drift) if self._ema_drift is not None else 0.0,
                "grad_base": float(self._last_grad_base),
                "grad_drift": float(self._last_grad_drift),
                "neg_same_count": float(self._last_same_count),
                "neg_other_count": float(self._last_other_count),
                "batch_unique_seed_count": float(seed_ids.unique().numel()),
                "n_neg_pool": float(drift_out.n_neg_pool),
                "n_neg_selected": float(drift_out.n_neg_selected),
                "selection_ratio": float(
                    drift_out.n_neg_selected / max(drift_out.n_neg_pool, 1)
                ),
                "hardness_min": float(drift_out.hardness_min),
                "hardness_mean": float(drift_out.hardness_mean),
                "hardness_p10": float(drift_out.hardness_p10),
                "hardness_p50": float(drift_out.hardness_p50),
                "hardness_p90": float(drift_out.hardness_p90),
                "s_feat": float(drift_out.s_feat),
                "s_drift_mean": float(drift_out.s_drift_mean),
                "mean_abs_v_pre": float(drift_out.mean_abs_v_pre),
                "time_x_gen_ms": float(t_xgen_ms),
                "time_neg_sampling_ms": float(t_neg_ms),
                "time_compute_v_ms": float(t_compute_v_ms),
            }
        )

    def _compute_drift_terms(
            self,
            x1: torch.Tensor,
            x_hat_10: torch.Tensor,
            base_with_aux: torch.Tensor,
            seed_ids: torch.Tensor | None,
            batch_size: int,
            device: torch.device,
            global_step: int,
            total_steps: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """Compute drift branch outputs and scaling."""
        progress = float(min(max(global_step / max(total_steps, 1), 0.0), 1.0))
        if seed_ids is None:
            seed_ids_tensor = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            seed_ids_tensor = seed_ids.to(device=device, dtype=torch.long)

        # One-step generation path with gradients (for drift loss).
        t_xgen_start = time.perf_counter()
        x_gen = x_hat_10
        t_xgen_ms = (time.perf_counter() - t_xgen_start) * 1000.0

        t_neg_start = time.perf_counter()
        y_neg = self._sample_drift_negatives(
            x_gen=x_gen,
            seed_ids=seed_ids_tensor,
            progress=progress,
            global_step=global_step,
        )
        t_neg_ms = (time.perf_counter() - t_neg_start) * 1000.0

        drift_out = drifting_loss_rawspace(
            x_gen=x_gen,
            y_pos=x1.detach(),
            y_neg=y_neg.detach(),
            temperatures=self.drift_temperatures,
            use_a6_norm=self.drift_use_a6_norm,
            a6_eps=self.drift_a6_eps,
            hardneg_k=self.drift_hardneg_k,
        )

        drift_loss = drift_out.loss
        params = self.trainable_parameters()
        lambda_eff = self._compute_lambda_eff(base_with_aux, drift_loss, params)
        lambda_eff = float(max(0.0, lambda_eff))
        self._last_lambda_eff = lambda_eff
        total_loss = base_with_aux + drift_loss * lambda_eff
        return total_loss, drift_out, seed_ids_tensor, t_xgen_ms, t_neg_ms

    def compute_loss(
        self,
        x1: torch.Tensor,
        seed_ids: torch.Tensor | None = None,
        global_step: int = 0,
        total_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Compute MeanFlow loss for a batch of target weights.

        Generates fresh noise each call (matching the reference MeanFlow
        implementation) and implements the MeanFlow Identity-based training:
            u_tgt = v - (t - r) * du/dt
        where du/dt is computed via JVP.

        No CFG is used since weight generation is unconditional.

        Args:
            x1: Target samples (trained weights) of shape (batch_size, dim).

        Returns:
            Tuple of (loss, mse_value) where loss is the adaptive L2 loss
            and mse_value is the raw MSE for logging.
        """
        batch_size = x1.shape[0]
        device = x1.device

        base_loss, mse_val = self._compute_base_terms(x1)
        x_hat_10, onestep_aux_loss = self._compute_onestep_aux_consistency(x1)
        base_with_aux = base_loss + self.onestep_lambda * onestep_aux_loss

        metrics: dict[str, float] = self._init_metrics(
            base_loss=base_loss,
            onestep_aux_loss=onestep_aux_loss,
            base_with_aux=base_with_aux,
        )

        if not self.use_drift_loss:
            return base_with_aux, mse_val, metrics

        total_loss, drift_out, seed_ids_tensor, t_xgen_ms, t_neg_ms = (
            self._compute_drift_terms(
                x1=x1,
                x_hat_10=x_hat_10,
                base_with_aux=base_with_aux,
                seed_ids=seed_ids,
                batch_size=batch_size,
                device=device,
                global_step=global_step,
                total_steps=total_steps,
            )
        )
        self._update_metrics_with_drift(
            metrics=metrics,
            total_loss=total_loss,
            drift_out=drift_out,
            lambda_eff=self._last_lambda_eff,
            seed_ids=seed_ids_tensor,
            t_xgen_ms=t_xgen_ms,
            t_neg_ms=t_neg_ms,
        )
        return total_loss, mse_val, metrics

    def train(
        self,
        n_iters: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        patience: int = 1000000,
        log_freq: int = 100,
    ) -> None:
        """Train the MeanFlow model.

        Args:
            n_iters: Number of training iterations.
            optimizer: Optimizer for model parameters.
            scheduler: Optional learning rate scheduler.
            patience: Early stopping patience (iterations without improvement).
            log_freq: Frequency of logging training progress.
        """
        self.model.train()
        last_loss = float("inf")
        patience_count = 0

        logger.info(f"Starting MeanFlow training for {n_iters} iterations")
        logger.info(
            f"  flow_ratio={self.flow_ratio}, "
            f"t1_ratio={self.t1_ratio}, "
            f"r0_ratio={self.r0_ratio}, "
            f"onestep_lambda={self.onestep_lambda}, "
            f"time_dist={self.time_dist}(mu={self.time_dist_mu}, "
            f"sigma={self.time_dist_sigma}), "
            f"loss_gamma={self.loss_gamma}"
        )

        for i in range(n_iters):
            # Sample target weights (noise is generated fresh inside compute_loss)
            batch = self._sample_from_loader(self.targetloader, is_source=False)
            if isinstance(batch, tuple):
                if len(batch) >= 3:
                    x1, seed_ids, _iter_pos = batch
                else:
                    x1, seed_ids = batch
            else:
                x1 = batch
                seed_ids = None

            # Set input_dim on first iteration
            if self.input_dim is None:
                self.input_dim = x1.shape[-1]

            # Compute loss (fresh noise generated internally)
            loss, mse_val, metrics = self.compute_loss(
                x1=x1,
                seed_ids=seed_ids,
                global_step=i,
                total_steps=n_iters,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at iteration {i}: {loss.item()}")
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # Track best model
            loss_value = loss.item()
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            # Early stopping
            if loss_value > last_loss:
                patience_count += 1
                if patience_count >= patience:
                    logger.info(f"Early stopping at iteration {i}")
                    break
            else:
                patience_count = 0

            last_loss = loss_value

            # Logging
            if i % log_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                if self.drift_auto_scale == "ema":
                    ema_text = (
                        f"({metrics['ema_base']:.6f}/{metrics['ema_drift']:.6f})"
                    )
                    grad_text = "N/A"
                elif self.drift_auto_scale == "grad":
                    ema_text = "N/A"
                    grad_text = (
                        f"({metrics['grad_base']:.6f}/{metrics['grad_drift']:.6f})"
                    )
                else:
                    ema_text = "N/A"
                    grad_text = "N/A"
                logger.info(
                    f"Iter {i}/{n_iters} | "
                    f"Total: {metrics['total_loss']:.6f} | "
                    f"Base: {metrics['base_loss']:.6f} | "
                    f"1StepAux: {metrics['onestep_aux_loss']:.3e} | "
                    f"Drift: {metrics['drift_loss']:.3e} | "
                    f"Mean|V|: {metrics['mean_abs_v']:.3e} | "
                    f"LambdaEff: {metrics['lambda_drift_eff']:.6f} | "
                    f"MSE: {mse_val.item():.6f} | "
                    f"EMA(base/drift)={ema_text} | "
                    f"Grad(base/drift)={grad_text} | "
                    f"Neg(same/other)=({int(metrics['neg_same_count'])}/{int(metrics['neg_other_count'])}) | "
                    f"HardNeg(pool/sel/ratio)=({int(metrics['n_neg_pool'])}/{int(metrics['n_neg_selected'])}/{metrics['selection_ratio']:.3f}) | "
                    f"Hardness(min/mean/p10/p50/p90)=({metrics['hardness_min']:.3f}/{metrics['hardness_mean']:.3f}/{metrics['hardness_p10']:.3f}/{metrics['hardness_p50']:.3f}/{metrics['hardness_p90']:.3f}) | "
                    f"A6(s_feat/s_drift)=({metrics['s_feat']:.3e}/{metrics['s_drift_mean']:.3e}) | "
                    f"Mean|V|pre: {metrics['mean_abs_v_pre']:.3e} | "
                    f"TimeMs(x_gen/neg/V)=({metrics['time_x_gen_ms']:.2f}/{metrics['time_neg_sampling_ms']:.2f}/{metrics['time_compute_v_ms']:.2f}) | "
                    f"Best: {self.best_loss:.6f} | LR: {lr:.2e}"
                )
                if self.use_drift_loss and self._last_sampling_summary:
                    logger.info(
                        "Drift sampling | "
                        + self._last_sampling_summary
                    )

        logger.info(f"Training complete. Best loss: {self.best_loss:.6f}")

    def generate(
        self,
        x0: torch.Tensor,
        n_steps: int = 5,
    ) -> torch.Tensor:
        """Generate samples using MeanFlow multi-step sampling.

        Uses the average velocity for efficient sampling:
            z_r = z_t - (t - r) * u(z_t, r, t)

        With n_steps=1, this becomes one-step generation:
            z_0 = z_1 - u(z_1, r=0, t=1)

        Args:
            x0: Source samples (noise) of shape (n_samples, dim).
            n_steps: Number of sampling steps. 1 = one-step generation.

        Returns:
            Generated samples of shape (n_samples, dim).
        """
        # Load best model if available
        if self.best_model_state is not None:
            current_state = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        z = x0.clone().to(self.device)
        batch_size = z.size(0)

        # Time steps from 1.0 -> 0.0
        t_vals = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)

        with torch.no_grad():
            for i in range(n_steps):
                t = torch.full(
                    (batch_size, 1), t_vals[i], device=self.device
                )
                r = torch.full(
                    (batch_size, 1), t_vals[i + 1], device=self.device
                )

                u = self.model(z, t, r)
                z = z - (t - r) * u

        # Restore model state
        if self.best_model_state is not None:
            self.model.load_state_dict(current_state)
        self.model.train()

        return z

    def generate_weights(
        self,
        n_samples: int,
        source_std: float = 0.01,
        n_steps: int = 5,
    ) -> torch.Tensor:
        """Generate new weight samples from random noise.

        Args:
            n_samples: Number of weight samples to generate.
            source_std: Standard deviation of source noise.
            n_steps: Number of sampling steps.

        Returns:
            Generated weights of shape (n_samples, input_dim).

        Raises:
            ValueError: If input_dim is not set.
        """
        if self.input_dim is None:
            raise ValueError(
                "input_dim must be set before generating weights. "
                "Set it manually or train the model first."
            )

        source_samples = (
            torch.randn(n_samples, self.input_dim, device=self.device)
            * source_std
        )
        return self.generate(source_samples, n_steps=n_steps)

    def _init_metrics(
        self,
        base_loss: torch.Tensor,
        onestep_aux_loss: torch.Tensor,
        base_with_aux: torch.Tensor,
    ) -> dict[str, float]:
        """Initialize metric dictionary before optional drift branch."""
        return {
            "base_loss": float(base_loss.detach().item()),
            "onestep_aux_loss": float(onestep_aux_loss.detach().item()),
            "drift_loss": 0.0,
            "total_loss": float(base_with_aux.detach().item()),
            "mean_abs_v": 0.0,
            "lambda_drift_eff": 0.0,
            "ema_base": float(self._ema_base) if self._ema_base is not None else 0.0,
            "ema_drift": float(self._ema_drift) if self._ema_drift is not None else 0.0,
            "grad_base": 0.0,
            "grad_drift": 0.0,
            "neg_same_count": 0.0,
            "neg_other_count": 0.0,
            "batch_unique_seed_count": 0.0,
            "n_neg_pool": 0.0,
            "n_neg_selected": 0.0,
            "selection_ratio": 0.0,
            "hardness_min": 0.0,
            "hardness_mean": 0.0,
            "hardness_p10": 0.0,
            "hardness_p50": 0.0,
            "hardness_p90": 0.0,
            "s_feat": 0.0,
            "s_drift_mean": 0.0,
            "mean_abs_v_pre": 0.0,
            "time_x_gen_ms": 0.0,
            "time_neg_sampling_ms": 0.0,
            "time_compute_v_ms": 0.0,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.best_model_state
            or self.model.state_dict(),
            "best_loss": self.best_loss,
            "input_dim": self.input_dim,
            "method": "meanflow",
            "source_std": self.source_std,
            "flow_ratio": self.flow_ratio,
            "t1_ratio": self.t1_ratio,
            "r0_ratio": self.r0_ratio,
            "onestep_lambda": self.onestep_lambda,
            "time_dist": self.time_dist,
            "time_dist_mu": self.time_dist_mu,
            "time_dist_sigma": self.time_dist_sigma,
            "loss_gamma": self.loss_gamma,
            "sigma": self.sigma,
            "use_drift_loss": self.use_drift_loss,
            "drift_lambda": self.drift_lambda,
            "drift_target_ratio": self.drift_target_ratio,
            "drift_auto_scale": self.drift_auto_scale,
            "drift_temperatures": self.drift_temperatures,
            "drift_neg_max_total": self.drift_neg_max_total,
            "drift_use_a6_norm": self.drift_use_a6_norm,
            "drift_a6_eps": self.drift_a6_eps,
            "drift_hardneg_k": self.drift_hardneg_k,
            "drift_ema_decay": self.drift_ema_decay,
            "drift_scale_eps": self.drift_scale_eps,
        }
        torch.save(checkpoint, path)
        logger.info(f"MeanFlow checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_model_state = checkpoint["model_state_dict"]
        self.best_loss = checkpoint["best_loss"]
        self.input_dim = checkpoint.get("input_dim")
        self.source_std = checkpoint.get("source_std", 0.01)
        self.flow_ratio = checkpoint.get("flow_ratio", 0.75)
        self.t1_ratio = checkpoint.get("t1_ratio", 0.0)
        self.r0_ratio = checkpoint.get("r0_ratio", 0.0)
        if self.flow_ratio + self.t1_ratio + self.r0_ratio > 1.0:
            logger.warning(
                "Loaded flow/t1/r0 ratios sum > 1.0; clamping t1/r0 to fit. "
                "flow_ratio=%.4f, t1_ratio=%.4f, r0_ratio=%.4f",
                float(self.flow_ratio),
                float(self.t1_ratio),
                float(self.r0_ratio),
            )
            remain = max(0.0, 1.0 - float(self.flow_ratio))
            self.t1_ratio = min(float(self.t1_ratio), remain)
            self.r0_ratio = min(float(self.r0_ratio), max(0.0, remain - self.t1_ratio))
        # Backward compatibility with older endpoint_lambda checkpoints.
        self.onestep_lambda = checkpoint.get(
            "onestep_lambda",
            checkpoint.get("endpoint_lambda", 1.0),
        )
        self.time_dist = checkpoint.get("time_dist", "uniform")
        self.time_dist_mu = checkpoint.get("time_dist_mu", -0.4)
        self.time_dist_sigma = checkpoint.get("time_dist_sigma", 1.0)
        self.loss_gamma = checkpoint.get("loss_gamma", 0.0)
        self.sigma = checkpoint.get("sigma", 0.001)
        self.use_drift_loss = checkpoint.get("use_drift_loss", False)
        self.drift_lambda = checkpoint.get("drift_lambda", 1.0)
        self.drift_target_ratio = checkpoint.get("drift_target_ratio", 0.1)
        self.drift_auto_scale = checkpoint.get("drift_auto_scale", "ema")
        self.drift_temperatures = checkpoint.get(
            "drift_temperatures", [0.02, 0.05, 0.2]
        )
        self.drift_neg_max_total = checkpoint.get("drift_neg_max_total", 256)
        self.drift_use_a6_norm = checkpoint.get("drift_use_a6_norm", True)
        self.drift_a6_eps = checkpoint.get("drift_a6_eps", 1e-6)
        self.drift_hardneg_k = checkpoint.get("drift_hardneg_k", 64)
        self.drift_ema_decay = checkpoint.get("drift_ema_decay", 0.99)
        self.drift_scale_eps = checkpoint.get("drift_scale_eps", 1e-8)

        logger.info(f"MeanFlow checkpoint loaded from {path}")
