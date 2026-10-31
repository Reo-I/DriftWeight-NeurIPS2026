#!/usr/bin/env python3
"""Evaluate original final.pt checkpoints on the dataset test split.

This is intended for paper-ready "Original" source-population scores.

Example:
    uv run scripts/evaluate_original_final.py \
        --dataset cifar10 \
        --arch resnet18 \
        --checkpoint-dir data/CIFAR10_resnet18/checkpoint
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
else:
    DataLoader = Any


def _load_runtime_dependencies() -> None:
    global MLP, ResNet18, ResNet20
    global get_cifar10_loaders, get_fashion_mnist_loaders, get_mnist_loaders
    global has_batchnorm, recalibrate_bn_stats, torch

    import torch as torch_module
    from src.data import (
        get_cifar10_loaders as get_cifar10_loaders_func,
        get_fashion_mnist_loaders as get_fashion_mnist_loaders_func,
        get_mnist_loaders as get_mnist_loaders_func,
    )
    from src.flowmatching.bn_recalibration import (
        has_batchnorm as has_batchnorm_func,
        recalibrate_bn_stats as recalibrate_bn_stats_func,
    )
    from src.models import (
        MLP as MLP_class,
        ResNet18 as ResNet18_func,
        ResNet20 as ResNet20_class,
    )

    torch = torch_module
    get_cifar10_loaders = get_cifar10_loaders_func
    get_fashion_mnist_loaders = get_fashion_mnist_loaders_func
    get_mnist_loaders = get_mnist_loaders_func
    has_batchnorm = has_batchnorm_func
    recalibrate_bn_stats = recalibrate_bn_stats_func
    MLP = MLP_class
    ResNet18 = ResNet18_func
    ResNet20 = ResNet20_class


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


def _default_data_root(dataset: str, arch: str) -> str | None:
    if dataset == "cifar10" and arch == "resnet18":
        return "data/CIFAR10_resnet18/raw"
    if dataset == "cifar10":
        return "data/CIFAR10/raw"
    return None


def _get_loaders(
    dataset: str,
    arch: str,
    batch_size: int,
    num_workers: int,
    data_root: str | None,
) -> tuple[DataLoader, DataLoader]:
    if dataset == "mnist":
        return get_mnist_loaders(batch_size=batch_size, num_workers=num_workers)
    if dataset == "fashion_mnist":
        return get_fashion_mnist_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
        )
    if dataset == "cifar10":
        return get_cifar10_loaders(
            batch_size=batch_size,
            data_root=data_root or _default_data_root(dataset, arch),
            num_workers=num_workers,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _seed_sort_key(seed_dir: Path) -> tuple[int, str]:
    match = re.search(r"seed(\d+)$", seed_dir.name)
    if match is None:
        return 10**12, seed_dir.name
    return int(match.group(1)), seed_dir.name


def _seed_id(seed_dir: Path | None) -> int | None:
    if seed_dir is None:
        return None
    match = re.search(r"seed(\d+)$", seed_dir.name)
    return int(match.group(1)) if match is not None else None


def _discover_weight_files(
    checkpoint_dir: Path,
    checkpoint_name: str,
    seed_start: int | None,
    seed_end: int | None,
    max_models: int | None,
) -> list[tuple[int | None, Path]]:
    root_final = checkpoint_dir / checkpoint_name
    if root_final.exists():
        return [(None, root_final)]

    seed_dirs = sorted(
        [path for path in checkpoint_dir.glob("seed*") if path.is_dir()],
        key=_seed_sort_key,
    )
    weight_files: list[tuple[int | None, Path]] = []
    for seed_dir in seed_dirs:
        seed = _seed_id(seed_dir)
        if seed is not None:
            if seed_start is not None and seed < seed_start:
                continue
            if seed_end is not None and seed > seed_end:
                continue

        path = seed_dir / checkpoint_name
        if path.exists():
            weight_files.append((seed, path))

        if max_models is not None and len(weight_files) >= max_models:
            break

    if not weight_files:
        raise FileNotFoundError(
            f"No {checkpoint_name} files found in {checkpoint_dir} "
            "or its seed*/ subdirectories."
        )
    return weight_files


def _load_state_dict(path: Path, device: torch.device) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{path} does not contain a PyTorch state_dict.")
    return checkpoint


def _evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
    model.eval().to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.size(0))

    return total_loss / total, 100.0 * correct / total, total


def _summarize(values: list[float], prefix: str) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        f"mean_{prefix}": mean,
        f"std_{prefix}": variance**0.5,
        f"min_{prefix}": min(values),
        f"max_{prefix}": max(values),
    }


def _write_stats(
    stats_path: Path,
    key: str,
    result: dict[str, Any],
    update_top_level: bool,
) -> None:
    if stats_path.exists():
        with stats_path.open() as f:
            stats = json.load(f)
    else:
        stats = {}

    stats[key] = result
    if update_top_level:
        previous_accuracy_keys = [
            "n_models",
            "accuracies",
            "mean_accuracy",
            "std_accuracy",
            "min_accuracy",
            "max_accuracy",
        ]
        if "best_accuracy_during_training" not in stats:
            previous = {
                name: stats[name] for name in previous_accuracy_keys if name in stats
            }
            if previous:
                stats["best_accuracy_during_training"] = previous

        stats["n_models"] = result["n_models"]
        stats["accuracies"] = result["accuracies"]
        stats["mean_accuracy"] = result["mean_accuracy"]
        stats["std_accuracy"] = result["std_accuracy"]
        stats["min_accuracy"] = result["min_accuracy"]
        stats["max_accuracy"] = result["max_accuracy"]
        stats["accuracy_source"] = "original_test_final_pt"

    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["mnist", "fashion_mnist", "cifar10"],
    )
    parser.add_argument(
        "--arch",
        default="resnet18",
        choices=["mlp", "resnet20", "resnet18"],
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/CIFAR10_resnet18/checkpoint"),
        help="Directory containing final.pt or seed*/final.pt files.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="final.pt",
        help="Checkpoint filename to evaluate inside each seed directory.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="JSON file to update (default: <checkpoint-dir>/training_stats.json).",
    )
    parser.add_argument(
        "--stats-key",
        default="original_test",
        help="Key written inside training_stats.json.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Default: cuda if available, else cpu.",
    )
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-end", type=int, default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument(
        "--no-bn-recalibration",
        action="store_true",
        help="Disable BatchNorm recalibration before evaluation.",
    )
    parser.add_argument(
        "--no-update-stats",
        action="store_true",
        help="Only print results; do not update training_stats.json.",
    )
    parser.add_argument(
        "--no-update-top-level",
        action="store_true",
        help=(
            "Do not replace top-level accuracy fields with final.pt test "
            "results. The nested stats key is still written."
        ),
    )
    parser.add_argument(
        "--skip-bad-checkpoints",
        action="store_true",
        help=(
            "Skip checkpoints that cannot be loaded or evaluated, and record "
            "them under skipped_checkpoints."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        _load_runtime_dependencies()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing runtime dependency: {exc.name}. Activate the project "
            "environment first (for example: source .venv/bin/activate)."
        ) from exc

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    stats_path = args.stats_path or (args.checkpoint_dir / "training_stats.json")
    recalibrate_bn = (
        args.dataset == "cifar10"
        and args.arch == "resnet18"
        and not args.no_bn_recalibration
    )

    print(f"Dataset/Arch: {args.dataset}/{args.arch}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Device: {device}")
    print(f"BatchNorm recalibration: {recalibrate_bn}")

    weight_files = _discover_weight_files(
        args.checkpoint_dir,
        args.checkpoint_name,
        args.seed_start,
        args.seed_end,
        args.max_models,
    )
    print(f"Found {len(weight_files)} checkpoint(s).")

    _, test_loader = _get_loaders(
        args.dataset,
        args.arch,
        args.batch_size,
        args.num_workers,
        args.data_root,
    )

    per_model: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    accuracies: list[float] = []
    losses: list[float] = []
    n_test_samples: int | None = None

    for index, (seed, path) in enumerate(weight_files, start=1):
        try:
            model = _build_model(args.dataset, args.arch)
            model.load_state_dict(_load_state_dict(path, device))
            model.to(device)

            if recalibrate_bn and has_batchnorm(model):
                model = recalibrate_bn_stats(
                    model,
                    calibration_loader=test_loader,
                    device=device,
                )

            loss, accuracy, total = _evaluate(model, test_loader, device)
        except Exception as exc:
            if not args.skip_bad_checkpoints:
                print(
                    f"[{index:>3}/{len(weight_files)}] "
                    f"seed={seed if seed is not None else 'root'} "
                    f"path={path} failed: {exc}",
                    file=sys.stderr,
                )
                raise

            skipped.append(
                {
                    "seed": seed,
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[{index:>3}/{len(weight_files)}] "
                f"seed={seed if seed is not None else 'root'} "
                f"skip: {type(exc).__name__}: {exc}"
            )
            continue

        n_test_samples = total
        accuracies.append(accuracy)
        losses.append(loss)
        per_model.append(
            {
                "seed": seed,
                "path": str(path),
                "test_loss": loss,
                "test_accuracy": accuracy,
            }
        )
        print(
            f"[{index:>3}/{len(weight_files)}] "
            f"seed={seed if seed is not None else 'root'} "
            f"acc={accuracy:.2f}% loss={loss:.4f}"
        )

    if not accuracies:
        raise ValueError("No checkpoints were evaluated successfully.")

    result = {
        "dataset": args.dataset,
        "arch": args.arch,
        "split": "test",
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_name": args.checkpoint_name,
        "n_models": len(accuracies),
        "n_discovered_checkpoints": len(weight_files),
        "n_skipped_checkpoints": len(skipped),
        "n_test_samples": n_test_samples,
        "batch_size": args.batch_size,
        "device": str(device),
        "bn_recalibrated": recalibrate_bn,
        "accuracies": accuracies,
        "losses": losses,
        **_summarize(accuracies, "accuracy"),
        **_summarize(losses, "loss"),
        "per_model": per_model,
        "skipped_checkpoints": skipped,
    }

    print("\nOriginal final.pt test summary")
    print("=" * 60)
    print(
        f"Accuracy: {result['mean_accuracy']:.2f}% "
        f"+/- {result['std_accuracy']:.2f}% "
        f"(min={result['min_accuracy']:.2f}%, max={result['max_accuracy']:.2f}%)"
    )
    print(
        f"Loss: {result['mean_loss']:.4f} "
        f"+/- {result['std_loss']:.4f}"
    )
    if skipped:
        print(f"Skipped checkpoints: {len(skipped)}")

    if not args.no_update_stats:
        _write_stats(
            stats_path,
            args.stats_key,
            result,
            update_top_level=not args.no_update_top_level,
        )
        print(f"Updated {stats_path} at key '{args.stats_key}'.")
        if not args.no_update_top_level:
            print("Updated top-level accuracy fields for paper reporting.")


if __name__ == "__main__":
    main()
