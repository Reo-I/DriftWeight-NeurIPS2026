#!/usr/bin/env python3
"""Compute MaxIoU (functional diversity) from an existing result directory.

The MaxIoU metric reported in the DriftWeight paper is defined as follows
(Section: "Main results: diversity"):

    For each generated model g, let W_g denote the set of test-sample
    indices mis-classified by g.  For each training model t, let W_t be
    the analogous set.  Then

        IoU(W_g, W_t) = |W_g ∩ W_t| / |W_g ∪ W_t|
        MaxIoU(g)    = max_t IoU(W_g, W_t)

    The number reported in the paper is mean_g MaxIoU(g).  A
    leave-one-out MaxIoU on the training set is used as the
    "diversity ceiling".

Usage examples
--------------
# MNIST / MLP (single result dir):
python -m scripts.compute_maxiou \
    --result-dir results/ablation_20260418_122750_27527_no_drift \
    --checkpoint-dir data/MNIST/checkpoint

# CIFAR-10 / ResNet-18 (enables BN recalibration automatically):
python -m scripts.compute_maxiou \
    --result-dir results/local_...your_cifar10_run... \
    --checkpoint-dir data/CIFAR10_resnet18/checkpoint

The script reads dataset/arch information from ``config.json`` inside
``--result-dir`` and auto-selects loaders accordingly.

Caveat
------
``generated_models/*.pt`` is overwritten across ``eval_steps`` during
training, so only the final step's weights are available after the run
finishes.  To compute MaxIoU for other step counts, re-generate weights
from ``flow_model.pt`` and ``pca.npz`` first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    get_cifar10_loaders,
    get_fashion_mnist_loaders,
    get_mnist_loaders,
)
from src.flowmatching.bn_recalibration import (  # noqa: E402
    has_batchnorm,
    recalibrate_bn_stats,
)
from src.flowmatching.weight_space import WeightSpaceObjectStateDict  # noqa: E402
from src.models import MLP, ResNet18, ResNet20  # noqa: E402


def _mlp_hidden_dim(dataset: str) -> int:
    if dataset == "mnist":
        return 32
    if dataset == "fashion_mnist":
        return 128
    raise ValueError(f"Unsupported MLP dataset: {dataset}")


def _build_model(dataset: str, arch: str) -> torch.nn.Module:
    if dataset in {"mnist", "fashion_mnist"} and arch == "mlp":
        return MLP(hidden_dim=_mlp_hidden_dim(dataset))
    if dataset == "cifar10" and arch == "resnet20":
        return ResNet20()
    if dataset == "cifar10" and arch == "resnet18":
        return ResNet18()
    raise ValueError(f"Unsupported dataset/arch: {dataset}/{arch}")


def _get_test_loader(
    dataset: str, arch: str, batch_size: int = 256
) -> tuple[DataLoader, DataLoader | None]:
    """Return (test_loader, calibration_loader). calibration_loader is None
    except for CIFAR-10/ResNet-18 (DeepWeightFlow-style BN recalibration)."""
    if dataset == "mnist":
        _, test = get_mnist_loaders(batch_size=batch_size)
        return test, None
    if dataset == "fashion_mnist":
        _, test = get_fashion_mnist_loaders(batch_size=batch_size)
        return test, None
    if dataset == "cifar10":
        cifar_root = (
            "data/CIFAR10_resnet18/raw" if arch == "resnet18" else "data/CIFAR10/raw"
        )
        _, test = get_cifar10_loaders(batch_size=batch_size, data_root=cifar_root)
        calibration = test if arch == "resnet18" else None
        return test, calibration
    raise ValueError(f"Unsupported dataset: {dataset}")


@torch.no_grad()
def _wrong_mask(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    """Return a boolean array of shape (n_test,) with True for wrong predictions.

    The test loader must be constructed with ``shuffle=False`` so the
    returned mask is aligned across different models.
    """
    model.eval().to(device)
    parts: list[np.ndarray] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        parts.append((preds != labels).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(bool)


def _masks_to_iou_matrix(
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """IoU between every row of ``a`` and every row of ``b``.

    Args:
        a: Boolean array of shape (n_a, n_samples).
        b: Boolean array of shape (n_b, n_samples).

    Returns:
        Float array of shape (n_a, n_b) where entry (i, j) is
        |a_i ∩ b_j| / |a_i ∪ b_j|.  Zero-division (both empty) yields 0.
    """
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    inter = a_f @ b_f.T                     # (n_a, n_b)
    size_a = a_f.sum(axis=1, keepdims=True)  # (n_a, 1)
    size_b = b_f.sum(axis=1, keepdims=True).T  # (1, n_b)
    union = size_a + size_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)


def _collect_training_wrong_masks(
    checkpoint_dir: Path,
    dataset: str,
    arch: str,
    test_loader: DataLoader,
    calibration_loader: DataLoader | None,
    device: torch.device,
    num_models: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    seed_dirs = sorted(checkpoint_dir.glob("seed*"))
    if num_models is not None:
        seed_dirs = seed_dirs[:num_models]
    if not seed_dirs:
        raise ValueError(f"No seed directories found in {checkpoint_dir}")

    masks: list[np.ndarray] = []
    seed_ids: list[int] = []
    for seed_dir in seed_dirs:
        final_path = seed_dir / "final.pt"
        if not final_path.exists():
            ckpts = sorted(seed_dir.glob("checkpoint_iter*.pt"))
            if not ckpts:
                print(f"  [skip] no checkpoint in {seed_dir}", file=sys.stderr)
                continue
            final_path = ckpts[-1]

        sd = torch.load(final_path, map_location=device)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        model = _build_model(dataset, arch)
        model.load_state_dict(sd)
        model.to(device)

        if calibration_loader is not None and has_batchnorm(model):
            model = recalibrate_bn_stats(
                model, calibration_loader=calibration_loader, device=str(device)
            )

        masks.append(_wrong_mask(model, test_loader, device))
        seed_ids.append(int(seed_dir.name.replace("seed", "")))
        if len(masks) % 10 == 0:
            print(f"  training models evaluated: {len(masks)}/{len(seed_dirs)}")

    if not masks:
        raise ValueError("No training models evaluated.")
    return np.stack(masks, axis=0), seed_ids


def _collect_generated_wrong_masks(
    generated_dir: Path,
    dataset: str,
    arch: str,
    test_loader: DataLoader,
    calibration_loader: DataLoader | None,
    device: torch.device,
) -> np.ndarray:
    files = sorted(generated_dir.glob("generated_model_*.pt"))
    if not files:
        raise ValueError(f"No generated_model_*.pt files under {generated_dir}")

    masks: list[np.ndarray] = []
    for i, path in enumerate(files):
        flat = torch.load(path, map_location=device)
        template = _build_model(dataset, arch)
        wso = WeightSpaceObjectStateDict.from_flat(flat, template, device=str(device))
        wso.apply_to_model(template)
        template.to(device)

        if calibration_loader is not None and has_batchnorm(template):
            template = recalibrate_bn_stats(
                template, calibration_loader=calibration_loader, device=str(device)
            )

        masks.append(_wrong_mask(template, test_loader, device))
        if (i + 1) % 10 == 0:
            print(f"  generated models evaluated: {i + 1}/{len(files)}")

    return np.stack(masks, axis=0)


def _loo_maxiou(masks: np.ndarray) -> np.ndarray:
    """Leave-one-out MaxIoU per row (diversity ceiling)."""
    iou = _masks_to_iou_matrix(masks, masks)
    np.fill_diagonal(iou, -np.inf)
    return iou.max(axis=1)


def _summarize(values: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_n": int(values.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Directory containing config.json + generated_models/",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Training-model checkpoint root (e.g. data/MNIST/checkpoint).",
    )
    parser.add_argument(
        "--num-training-models",
        type=int,
        default=None,
        help="If set, only the first N training models are used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (default: auto cuda/mps/cpu).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Test-time batch size."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <result-dir>/maxiou_results.json).",
    )
    args = parser.parse_args()

    result_dir: Path = args.result_dir
    config_path = result_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {result_dir}")
    with config_path.open() as f:
        config = json.load(f)
    dataset = config["dataset"]
    arch = config["arch"]

    generated_dir = result_dir / "generated_models"
    if not generated_dir.exists():
        raise FileNotFoundError(
            f"{generated_dir} not found. "
            "Re-run training with --save-generated-models (default True) "
            "or regenerate from flow_model.pt + pca.npz."
        )

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # NOTE: MPS is intentionally avoided here. Small MLP fc layers (e.g.
        # hidden=32 -> 10) can produce numerically different argmax outputs
        # under MPS, which corrupts wrong-mask sets and inflates MaxIoU.
        # CPU evaluation of 100 MNIST-MLP models over 10k test samples
        # finishes in ~20s, so there is no meaningful speed loss.
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "mps":
        print(
            "WARNING: MPS backend can produce inconsistent argmax for small "
            "fully-connected classifiers. Pass --device cpu for paper-grade "
            "numbers."
        )
    print(f"Dataset/Arch: {dataset}/{arch}")

    test_loader, calibration_loader = _get_test_loader(
        dataset, arch, batch_size=args.batch_size
    )

    print(f"\n[1/2] Evaluating training models from {args.checkpoint_dir} ...")
    train_masks, _ = _collect_training_wrong_masks(
        args.checkpoint_dir,
        dataset,
        arch,
        test_loader,
        calibration_loader,
        device,
        num_models=args.num_training_models,
    )
    print(
        f"  training models: {train_masks.shape[0]}, "
        f"mean wrong count = {train_masks.sum(1).mean():.1f}"
    )

    print(f"\n[2/2] Evaluating generated models from {generated_dir} ...")
    gen_masks = _collect_generated_wrong_masks(
        generated_dir, dataset, arch, test_loader, calibration_loader, device
    )
    print(
        f"  generated models: {gen_masks.shape[0]}, "
        f"mean wrong count = {gen_masks.sum(1).mean():.1f}"
    )

    iou = _masks_to_iou_matrix(gen_masks, train_masks)  # (n_gen, n_train)
    maxiou_per_gen = iou.max(axis=1)
    ceiling = _loo_maxiou(train_masks)

    summary = {
        "result_dir": str(result_dir),
        "dataset": dataset,
        "arch": arch,
        "n_generated": int(gen_masks.shape[0]),
        "n_training": int(train_masks.shape[0]),
        "n_test_samples": int(train_masks.shape[1]),
        **_summarize(maxiou_per_gen, "maxiou_generated"),
        **_summarize(ceiling, "maxiou_training_ceiling_loo"),
        "maxiou_per_generated": maxiou_per_gen.tolist(),
        "maxiou_training_loo": ceiling.tolist(),
    }

    out_path = args.output or (result_dir / "maxiou_results.json")
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("MaxIoU summary")
    print("=" * 60)
    print(
        f"  Generated models : mean={summary['maxiou_generated_mean']:.4f}  "
        f"std={summary['maxiou_generated_std']:.4f}  "
        f"min={summary['maxiou_generated_min']:.4f}  "
        f"max={summary['maxiou_generated_max']:.4f}"
    )
    print(
        f"  Training ceiling : mean={summary['maxiou_training_ceiling_loo_mean']:.4f}  "
        f"std={summary['maxiou_training_ceiling_loo_std']:.4f}  "
        f"(leave-one-out)"
    )
    print("=" * 60)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
