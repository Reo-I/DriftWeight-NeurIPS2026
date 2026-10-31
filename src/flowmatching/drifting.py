"""Drifting loss components for MeanFlow/FlowMatching training.

This module implements the core pieces required by the Drifting Models style
training objective:
  - compute_V (Algorithm 2 style doubly-normalized drift field)
  - feature and drift normalization
  - seeded trajectory replay buffer for negative samples
  - multi-scale feature encoder phi
  - feature-space drifting loss aggregation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _sync_for_timing(device: torch.device) -> None:
    """Synchronize device for more accurate wall-clock timing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def _now(device: torch.device) -> float:
    _sync_for_timing(device)
    return time.perf_counter()


def compute_V(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    T: float,
    y_neg_includes_x: bool = False,
) -> torch.Tensor:
    """Compute drifting vector field V using doubly-normalized softmax weights.

    Args:
        x: Query vectors of shape (N, d).
        y_pos: Positive vectors of shape (N_pos, d).
        y_neg: Negative vectors of shape (N_neg, d).
        T: Temperature scalar.
        y_neg_includes_x: If True, y_neg starts with x in the same order.

    Returns:
        V tensor of shape (N, d).
    """
    if x.ndim != 2 or y_pos.ndim != 2 or y_neg.ndim != 2:
        raise ValueError("compute_V expects rank-2 tensors for x/y_pos/y_neg")
    if x.size(1) != y_pos.size(1) or x.size(1) != y_neg.size(1):
        raise ValueError("Feature dimension mismatch in compute_V inputs")
    if T <= 0.0:
        raise ValueError(f"Temperature must be positive, got {T}")

    _, y_pos32, y_neg32, dist_pos, dist_neg = _prepare_pairwise_distances(
        x, y_pos, y_neg, y_neg_includes_x=y_neg_includes_x
    )
    return _compute_V_from_distances(
        y_pos32=y_pos32,
        y_neg32=y_neg32,
        dist_pos=dist_pos,
        dist_neg=dist_neg,
        T=T,
        out_dtype=x.dtype,
    )


def _prepare_pairwise_distances(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    y_neg_includes_x: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare fp32 pairwise L2 distances once for multi-temperature reuse."""
    if x.ndim != 2 or y_pos.ndim != 2 or y_neg.ndim != 2:
        raise ValueError("Expected rank-2 tensors for x/y_pos/y_neg")
    if x.size(1) != y_pos.size(1) or x.size(1) != y_neg.size(1):
        raise ValueError("Feature dimension mismatch")

    x32 = x.float()
    y_pos32 = y_pos.float()
    y_neg32 = y_neg.float()

    dist_pos = torch.cdist(x32, y_pos32, p=2)  # (N, N_pos)
    dist_neg = torch.cdist(x32, y_neg32, p=2)  # (N, N_neg)

    if y_neg_includes_x:
        n = x.size(0)
        if y_neg.size(0) < n:
            raise ValueError(
                "y_neg_includes_x=True requires y_neg size >= batch size"
            )
        eye = torch.eye(n, device=dist_neg.device, dtype=dist_neg.dtype)
        dist_neg = dist_neg.clone()
        dist_neg[:, :n] = dist_neg[:, :n] + eye * 1e6

    return x32, y_pos32, y_neg32, dist_pos, dist_neg


def _compute_V_from_distances(
    y_pos32: torch.Tensor,
    y_neg32: torch.Tensor,
    dist_pos: torch.Tensor,
    dist_neg: torch.Tensor,
    T: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute V from precomputed distances for one temperature."""
    if T <= 0.0:
        raise ValueError(f"Temperature must be positive, got {T}")
    logit_pos = -dist_pos / float(T)
    logit_neg = -dist_neg / float(T)
    logit = torch.cat([logit_pos, logit_neg], dim=1)

    # Doubly-normalized softmax (geometric mean)
    A_row = F.softmax(logit, dim=-1)
    A_col = F.softmax(logit, dim=-2)
    A = torch.sqrt(torch.clamp(A_row * A_col, min=1e-30))

    n_pos = y_pos32.size(0)
    A_pos = A[:, :n_pos]
    A_neg = A[:, n_pos:]

    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    drift_pos = W_pos @ y_pos32
    drift_neg = W_neg @ y_neg32
    V = drift_pos - drift_neg
    return V.to(dtype=out_dtype)


def feature_normalize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize feature vectors along channel dimension."""
    if features.ndim != 3:
        raise ValueError("feature_normalize expects shape (N, M, C)")
    norm = torch.sqrt(torch.clamp((features ** 2).mean(dim=-1, keepdim=True), min=eps))
    return features / norm


def drift_normalize(drift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize drift vectors for scale-robust temperature usage."""
    if drift.ndim != 2:
        raise ValueError("drift_normalize expects shape (N, C)")
    scale = torch.sqrt(torch.clamp((drift ** 2).mean(dim=-1, keepdim=True), min=eps))
    return drift / scale


@dataclass
class DriftLossOutput:
    loss: torch.Tensor
    num_feature_groups: int
    compute_v_seconds: float = 0.0
    mean_abs_v: float = 0.0
    mean_abs_v_pre: float = 0.0
    s_feat: float = 1.0
    s_drift_mean: float = 1.0
    n_neg_pool: int = 0
    n_neg_selected: int = 0
    hardness_min: float = 0.0
    hardness_mean: float = 0.0
    hardness_p10: float = 0.0
    hardness_p50: float = 0.0
    hardness_p90: float = 0.0


def _hardness_summary(hardness: torch.Tensor) -> tuple[float, float, float, float, float]:
    if hardness.numel() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    h = hardness.float()
    q = torch.tensor([0.1, 0.5, 0.9], device=h.device, dtype=h.dtype)
    p10, p50, p90 = torch.quantile(h, q).tolist()
    return (
        float(h.min().item()),
        float(h.mean().item()),
        float(p10),
        float(p50),
        float(p90),
    )


def _select_hard_negative_indices(
    dist_neg_norm: torch.Tensor,
    mandatory_count: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k hard negatives while preserving mandatory prefix order.

    When k <= 0, all negatives are returned (no filtering).
    """
    n_neg = dist_neg_norm.size(1)
    device = dist_neg_norm.device
    mandatory_count = max(0, min(int(mandatory_count), n_neg))
    mandatory_idx = torch.arange(mandatory_count, device=device, dtype=torch.long)

    # h_j = min_i dist(i, j): smaller => harder negative.
    hardness = dist_neg_norm.min(dim=0).values
    if k <= 0 or n_neg <= mandatory_count:
        return torch.arange(n_neg, device=device, dtype=torch.long), hardness

    cand_idx = torch.arange(mandatory_count, n_neg, device=device, dtype=torch.long)
    h_cand = hardness[cand_idx]
    if h_cand.numel() == 0:
        return mandatory_idx, hardness

    kk = min(max(int(k), 0), h_cand.numel())
    order = torch.argsort(h_cand)
    selected = cand_idx[order[:kk]]

    if selected.numel() > 0:
        selected = torch.unique(selected, sorted=False)
    final_idx = (
        torch.cat([mandatory_idx, selected], dim=0)
        if mandatory_idx.numel() > 0
        else selected
    )
    return final_idx, hardness


def _drift_loss_single_group(
        feat_x: torch.Tensor,
        feat_pos: torch.Tensor,
        feat_neg_pool: torch.Tensor,
        temperatures: list[float],
        use_a6_norm: bool,
        a6_eps: float,
        hardneg_k: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one drift-loss group with A.6 norm + hard negative."""
    with torch.no_grad():
        feat_pos_det = feat_pos.detach()
        feat_neg_det = feat_neg_pool.detach()

    # A.6 feature normalization scale (distance mean => 1).
    def _compute_feature_scale() -> torch.Tensor:
        d_x = feat_x.detach().float()
        d_pos = feat_pos_det.float()
        d_neg = feat_neg_det.float()
        dist_pos_raw = torch.cdist(d_x, d_pos, p=2)
        dist_neg_raw = torch.cdist(d_x, d_neg, p=2)
        return torch.cat([dist_pos_raw.reshape(-1), dist_neg_raw.reshape(-1)]).mean()

    s_feat_val = _compute_feature_scale()
    if use_a6_norm:
        s_feat = torch.clamp(s_feat_val.detach(), min=float(a6_eps))
    else:
        s_feat = feat_x.new_tensor(1.0)

    feat_x_n = feat_x / s_feat
    feat_pos_n = feat_pos_det / s_feat
    feat_neg_n = feat_neg_det / s_feat

    # Hard-negative selection on normalized feature distances.
    def _prepare_hardneg_inputs() -> tuple[torch.Tensor, int, int, float, float, float, float, float]:
        dist_neg_norm = torch.cdist(
            feat_x_n.detach().float(),
            feat_neg_n.detach().float(),
            p=2,
        )
        mandatory_n = min(feat_x.size(0), feat_neg_n.size(0))
        final_idx, hardness = _select_hard_negative_indices(
            dist_neg_norm=dist_neg_norm,
            mandatory_count=mandatory_n,
            k=hardneg_k,
        )
        feat_neg_hard_n = feat_neg_n[final_idx]
        n_neg_pool = int(feat_neg_n.size(0))
        n_neg_selected = int(final_idx.numel())
        h_min, h_mean, h_p10, h_p50, h_p90 = _hardness_summary(hardness)
        return (
            feat_neg_hard_n,
            n_neg_pool,
            n_neg_selected,
            h_min,
            h_mean,
            h_p10,
            h_p50,
            h_p90,
        )

    def _aggregate_v(
        feat_neg_hard_n: torch.Tensor,
        n_neg_selected: int,
    ) -> tuple[torch.Tensor, float, float, float, int]:
        # Precompute distances once for all temperatures.
        y_neg_includes_x = n_neg_selected >= feat_x_n.size(0)
        _, feat_pos32, feat_neg32, dist_pos, dist_neg = _prepare_pairwise_distances(
            feat_x_n.detach(), feat_pos_n.detach(), feat_neg_hard_n.detach(),
            y_neg_includes_x=y_neg_includes_x,
        )

        V_agg = torch.zeros_like(feat_x_n)
        abs_v_pre_acc = 0.0
        abs_v_post_acc = 0.0
        s_drift_acc = 0.0
        t_count = 0
        for temp in temperatures:
            V = _compute_V_from_distances(
                y_pos32=feat_pos32,
                y_neg32=feat_neg32,
                dist_pos=dist_pos,
                dist_neg=dist_neg,
                T=float(temp),
                out_dtype=feat_x_n.dtype,
            )
            abs_v_pre_acc += float(V.abs().mean().item())
            if use_a6_norm:
                s_drift = torch.clamp(
                    V.norm(dim=-1).mean().detach(),
                    min=float(a6_eps),
                )
                Vn = V / s_drift
            else:
                s_drift = V.new_tensor(1.0)
                Vn = V
            abs_v_post_acc += float(Vn.abs().mean().item())
            s_drift_acc += float(s_drift.item())
            t_count += 1
            V_agg = V_agg + Vn
        return V_agg, abs_v_pre_acc, abs_v_post_acc, s_drift_acc, t_count

    with torch.no_grad():
        (
            feat_neg_hard_n,
            n_neg_pool,
            n_neg_selected,
            h_min,
            h_mean,
            h_p10,
            h_p50,
            h_p90,
        ) = _prepare_hardneg_inputs()
        (
            V_agg,
            abs_v_pre_acc,
            abs_v_post_acc,
            s_drift_acc,
            t_count,
        ) = _aggregate_v(
            feat_neg_hard_n=feat_neg_hard_n,
            n_neg_selected=n_neg_selected,
        )

    target = (feat_x_n + V_agg).detach()
    loss = F.mse_loss(feat_x_n, target)
    stats = {
        "mean_abs_v_pre": abs_v_pre_acc / max(t_count, 1),
        "mean_abs_v": abs_v_post_acc / max(t_count, 1),
        "s_feat": float(s_feat.item()) if torch.is_tensor(s_feat) else float(s_feat),
        "s_drift_mean": s_drift_acc / max(t_count, 1),
        "n_neg_pool": float(n_neg_pool),
        "n_neg_selected": float(n_neg_selected),
        "hardness_min": h_min,
        "hardness_mean": h_mean,
        "hardness_p10": h_p10,
        "hardness_p50": h_p50,
        "hardness_p90": h_p90,
    }
    return loss, stats


def drifting_loss_rawspace(
    x_gen: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: list[float],
    use_a6_norm: bool = True,
    a6_eps: float = 1e-6,
    hardneg_k: int = 64,
) -> DriftLossOutput:
    """Compute drifting loss directly in raw weight space.

    Distance computation uses Euclidean (L2) distance in parameter space,
    matching the original drifting formulation.

    Args:
        x_gen: Generated samples (N, D).
        y_pos: Positive samples (N_pos, D).
        y_neg: Negative samples (N_neg, D).
        temperatures: Temperature list for compute_V.
    """
    start_v = _now(x_gen.device)

    # Keep current feature map behavior, then apply A.6 / hard-neg on top.
    f_x = feature_normalize(x_gen.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        f_pos = feature_normalize(y_pos.unsqueeze(1)).squeeze(1)
        f_neg = feature_normalize(y_neg.unsqueeze(1)).squeeze(1)

    loss, stats = _drift_loss_single_group(
        feat_x=f_x,
        feat_pos=f_pos,
        feat_neg_pool=f_neg,
        temperatures=temperatures,
        use_a6_norm=use_a6_norm,
        a6_eps=a6_eps,
        hardneg_k=hardneg_k,
    )
    compute_v_seconds = _now(x_gen.device) - start_v

    return DriftLossOutput(
        loss=loss,
        num_feature_groups=1,
        compute_v_seconds=compute_v_seconds,
        mean_abs_v=float(stats["mean_abs_v"]),
        mean_abs_v_pre=float(stats["mean_abs_v_pre"]),
        s_feat=float(stats["s_feat"]),
        s_drift_mean=float(stats["s_drift_mean"]),
        n_neg_pool=int(stats["n_neg_pool"]),
        n_neg_selected=int(stats["n_neg_selected"]),
        hardness_min=float(stats["hardness_min"]),
        hardness_mean=float(stats["hardness_mean"]),
        hardness_p10=float(stats["hardness_p10"]),
        hardness_p50=float(stats["hardness_p50"]),
        hardness_p90=float(stats["hardness_p90"]),
    )
