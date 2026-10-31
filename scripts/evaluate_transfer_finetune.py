#!/usr/bin/env python3
"""Evaluate CIFAR-10 ResNet-18 weights as transfer fine-tuning initializers.

By default this script follows the released DeepWeightFlow transfer-learning
protocol: initialize a ResNet-18 from generated, original CIFAR-10, or random
weights; use target-dataset normalization; fine-tune with AdamW; and report
zero-shot plus best-so-far short fine-tuning accuracy on STL-10 or SVHN.
Generated weights receive CIFAR-10 BatchNorm recalibration before transfer.

Examples
--------
python -m scripts.evaluate_transfer_finetune \
    --init-source original \
    --checkpoint-dir data/CIFAR10_resnet18/checkpoint \
    --target-dataset svhn \
    --epochs 0 1 5 \
    --n-models 5

python -m scripts.evaluate_transfer_finetune \
    --init-source generated \
    --result-dir results/cifar10_resnet18 \
    --checkpoint-dir data/CIFAR10_resnet18/checkpoint \
    --target-dataset stl10 \
    --n-models 5
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flowmatching import FlowMatching, WeightSpaceFlowModel  # noqa: E402
from src.flowmatching.artifacts import (  # noqa: E402
    generated_models_dir,
    resolve_generated_models_dir,
)
from src.flowmatching.bn_recalibration import (  # noqa: E402
    has_batchnorm,
    recalibrate_bn_stats,
)
from src.flowmatching.mean_flow import WeightSpaceMeanFlow  # noqa: E402
from src.flowmatching.meanflow_model import WeightSpaceMeanFlowModel  # noqa: E402
from src.flowmatching.pca import WeightSpacePCA  # noqa: E402
from src.flowmatching.weight_space import WeightSpaceObjectStateDict  # noqa: E402
from src.models import ResNet18  # noqa: E402


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2241, 0.2215, 0.2239)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

# STL-10 order: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
# CIFAR-10 order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
# Monkey has no CIFAR-10 counterpart; we keep output index 6 as the target class
# for fine-tuning, analogous to replacing the unused "frog" logit.
STL10_TO_CIFAR10 = {
    0: 0,
    1: 2,
    2: 1,
    3: 3,
    4: 4,
    5: 5,
    6: 7,
    7: 6,
    8: 8,
    9: 9,
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _target_normalization(
    target_dataset: str,
    protocol: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if protocol == "cifar_warmstart":
        return CIFAR10_MEAN, CIFAR10_STD
    if target_dataset == "stl10":
        return STL10_MEAN, STL10_STD
    if target_dataset == "svhn":
        return SVHN_MEAN, SVHN_STD
    raise ValueError(f"Unsupported target dataset: {target_dataset}")


def _target_transforms(target_dataset: str, train: bool, protocol: str) -> transforms.Compose:
    mean, std = _target_normalization(target_dataset, protocol)
    steps: list[Any] = [transforms.Resize((32, 32))]
    if train:
        steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(steps)


def _map_stl10_to_cifar10(label: int) -> int:
    return STL10_TO_CIFAR10[int(label)]


def _map_svhn_label(label: int) -> int:
    # Some SVHN readers use 10 for digit zero; torchvision normally normalizes it.
    label = int(label)
    return 0 if label == 10 else label


def _get_target_loaders(
    target_dataset: str,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    stl10_label_space: str,
    protocol: str,
) -> tuple[DataLoader, DataLoader]:
    if target_dataset == "stl10":
        target_transform = (
            _map_stl10_to_cifar10 if stl10_label_space == "cifar10" else None
        )
        train_dataset = datasets.STL10(
            root=str(data_root / "STL10"),
            split="train",
            download=True,
            transform=_target_transforms(target_dataset, train=True, protocol=protocol),
            target_transform=target_transform,
        )
        test_dataset = datasets.STL10(
            root=str(data_root / "STL10"),
            split="test",
            download=True,
            transform=_target_transforms(target_dataset, train=False, protocol=protocol),
            target_transform=target_transform,
        )
    elif target_dataset == "svhn":
        train_dataset = datasets.SVHN(
            root=str(data_root / "SVHN"),
            split="train",
            download=True,
            transform=_target_transforms(target_dataset, train=True, protocol=protocol),
            target_transform=_map_svhn_label,
        )
        test_dataset = datasets.SVHN(
            root=str(data_root / "SVHN"),
            split="test",
            download=True,
            transform=_target_transforms(target_dataset, train=False, protocol=protocol),
            target_transform=_map_svhn_label,
        )
    else:
        raise ValueError(f"Unsupported target dataset: {target_dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def _seed_sort_key(seed_dir: Path) -> tuple[int, str]:
    match = re.search(r"seed(\d+)$", seed_dir.name)
    if match is None:
        return 10**12, seed_dir.name
    return int(match.group(1)), seed_dir.name


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} does not contain a state_dict-like object.")
    return checkpoint


def _discover_original_checkpoints(
    checkpoint_dir: Path,
    n_models: int,
    seed_start: int | None,
    seed_end: int | None,
) -> list[Path]:
    seed_dirs = sorted(
        [path for path in checkpoint_dir.glob("seed*") if path.is_dir()],
        key=_seed_sort_key,
    )
    paths: list[Path] = []
    for seed_dir in seed_dirs:
        seed_match = re.search(r"seed(\d+)$", seed_dir.name)
        seed = int(seed_match.group(1)) if seed_match else None
        if seed is not None:
            if seed_start is not None and seed < seed_start:
                continue
            if seed_end is not None and seed > seed_end:
                continue
        final_path = seed_dir / "final.pt"
        if final_path.exists():
            paths.append(final_path)
        if len(paths) >= n_models:
            break
    if len(paths) < n_models:
        raise FileNotFoundError(
            f"Found {len(paths)} final.pt files under {checkpoint_dir}; "
            f"need {n_models}."
        )
    return paths


def _model_from_flat(flat: torch.Tensor, device: torch.device) -> torch.nn.Module:
    model = ResNet18()
    wso = WeightSpaceObjectStateDict.from_flat(flat, model, device=device)
    wso.apply_to_model(model)
    return model.to(device)


def _generated_files(result_dir: Path, n_steps: int = 1) -> list[Path]:
    generated_dir = resolve_generated_models_dir(result_dir, n_steps)
    return sorted(generated_dir.glob("generated_model_*.pt"))


def _load_config(result_dir: Path) -> dict[str, Any]:
    config_path = result_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {result_dir}")
    with config_path.open() as f:
        return json.load(f)


def _regenerate_generated_flats(
    result_dir: Path,
    n_models: int,
    n_steps: int,
    device: torch.device,
    save_generated: bool,
) -> list[torch.Tensor]:
    config = _load_config(result_dir)
    flow_path = result_dir / "flow_model.pt"
    pca_path = result_dir / "pca.npz"
    missing = [str(path) for path in (flow_path, pca_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot regenerate generated weights because required artifacts are "
            f"missing: {', '.join(missing)}. Re-run the DriftWeight job with "
            "--save-generated-models or keep flow_model.pt and pca.npz."
        )
    empty = [str(path) for path in (flow_path, pca_path) if path.stat().st_size == 0]
    if empty:
        raise FileNotFoundError(
            "Cannot regenerate generated weights because required artifacts are "
            f"empty/corrupt: {', '.join(empty)}. Re-run the DriftWeight job with "
            "--save-generated-models or keep non-empty flow_model.pt and pca.npz."
        )
    if not zipfile.is_zipfile(pca_path):
        raise FileNotFoundError(
            "Cannot regenerate generated weights because pca.npz is not a valid "
            f"NumPy .npz archive: {pca_path}. The file may be partially copied "
            "or truncated; re-copy it before running this evaluation."
        )

    actual_dim = int(config.get("actual_dim") or config.get("pca_components"))
    hidden_dim = int(config.get("hidden_dim", 512))
    time_embed_dim = int(config.get("time_embed_dim", 64))
    dropout = float(config.get("dropout", 0.1))
    source_std = float(config.get("source_std", 0.01))
    method = config.get("method", "meanflow")

    dummy_loader = DataLoader(
        TensorDataset(torch.zeros(1, actual_dim), torch.zeros(1, dtype=torch.long)),
        batch_size=1,
    )
    if method == "meanflow":
        model = WeightSpaceMeanFlowModel(
            input_dim=actual_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        )
        trainer = WeightSpaceMeanFlow(
            targetloader=dummy_loader,
            model=model,
            device=device,
            source_std=source_std,
            sigma=float(config.get("sigma", 0.001)),
        )
        trainer.load_checkpoint(str(flow_path))
        latent = trainer.generate_weights(
            n_samples=n_models,
            source_std=source_std,
            n_steps=n_steps,
        )
    elif method == "flowmatching":
        model = WeightSpaceFlowModel(
            input_dim=actual_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        )
        trainer = FlowMatching(
            targetloader=dummy_loader,
            model=model,
            device=device,
            source_std=source_std,
            sigma=float(config.get("sigma", 0.001)),
            t_dist=config.get("t_dist", "uniform"),
        )
        trainer.load_checkpoint(str(flow_path))
        latent = trainer.generate_weights(
            n_samples=n_models,
            source_std=source_std,
            n_steps=n_steps,
            method=config.get("integration_method", "rk4"),
        )
    else:
        raise ValueError(f"Unsupported method in config: {method}")

    pca = WeightSpacePCA.load(str(pca_path))
    flats = list(pca.inverse_transform(latent, device=str(device)))
    if save_generated:
        generated_dir = generated_models_dir(result_dir, n_steps)
        generated_dir.mkdir(parents=True, exist_ok=True)
        for idx, flat in enumerate(flats):
            torch.save(flat.detach().cpu(), generated_dir / f"generated_model_{idx:04d}.pt")
    return flats


def _load_initial_models(args: argparse.Namespace, device: torch.device) -> list[torch.nn.Module]:
    models: list[torch.nn.Module] = []
    if args.init_source == "random":
        for i in range(args.n_models):
            torch.manual_seed(args.seed + i)
            models.append(ResNet18().to(device))
        return models

    if args.init_source == "original":
        paths = _discover_original_checkpoints(
            args.checkpoint_dir,
            n_models=args.n_models,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
        )
        for path in paths:
            model = ResNet18()
            model.load_state_dict(_load_state_dict(path, device))
            models.append(model.to(device))
        return models

    if args.init_source == "generated":
        if args.result_dir is None:
            raise ValueError("--result-dir is required for --init-source generated")
        files = _generated_files(args.result_dir, args.generated_steps)
        if files:
            for path in files[: args.n_models]:
                flat = torch.load(path, map_location=device)
                models.append(_model_from_flat(flat, device))
            if len(models) < args.n_models:
                raise FileNotFoundError(
                    f"Found {len(models)} generated models under "
                    f"{args.result_dir / 'generated_models'}; need {args.n_models}."
                )
            return models

        flats = _regenerate_generated_flats(
            args.result_dir,
            n_models=args.n_models,
            n_steps=args.generated_steps,
            device=device,
            save_generated=args.save_regenerated_models,
        )
        return [_model_from_flat(flat, device) for flat in flats]

    raise ValueError(f"Unsupported init source: {args.init_source}")


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval().to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, labels).item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return total_loss / total, 100.0 * correct / total


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.train().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


def _get_cifar10_bn_loader(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    source_data_root: Path | None,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    candidates = [
        source_data_root,
        Path("data/CIFAR10_resnet18/raw"),
        Path("data/CIFAR10/raw"),
        data_root / "CIFAR10",
    ]
    root = next(
        (
            path
            for path in candidates
            if path is not None and (path / "cifar-10-batches-py").exists()
        ),
        source_data_root or (data_root / "CIFAR10"),
    )
    dataset = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _freeze_feature_prefix(model: torch.nn.Module, detach_ratio: float) -> None:
    if detach_ratio <= 0:
        return
    layers = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]
    num_to_freeze = int(len(layers) * detach_ratio)
    for layer in layers[:num_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False


def _resolve_protocol_defaults(args: argparse.Namespace) -> dict[str, Any]:
    is_deepweightflow = args.protocol == "deepweightflow"
    optimizer_name = args.optimizer
    if optimizer_name == "auto":
        optimizer_name = "adamw" if is_deepweightflow else "sgd"

    lr = args.lr
    if lr is None:
        lr = 1e-4 if is_deepweightflow else 0.01
    weight_decay = args.weight_decay
    if weight_decay is None:
        weight_decay = 1e-5 if is_deepweightflow else 5e-4

    detach_ratio = args.detach_ratio
    if detach_ratio is None:
        detach_ratio = 0.0 if args.init_source == "random" else (0.4 if is_deepweightflow else 0.0)

    stl10_label_space = args.stl10_label_space
    if stl10_label_space == "auto":
        stl10_label_space = "native" if is_deepweightflow else "cifar10"

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 32 if is_deepweightflow else 128

    return {
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "detach_ratio": detach_ratio,
        "stl10_label_space": stl10_label_space,
        "batch_size": batch_size,
        "report_best": args.report_best or is_deepweightflow,
        "target_bn_recalibration": args.target_bn_recalibration
        or (args.recalibrate_bn and not is_deepweightflow),
        "source_bn_recalibration": args.source_bn_recalibration and is_deepweightflow,
    }


def _build_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    params = filter(lambda param: param.requires_grad, model.parameters())
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _summarize(values: Iterable[float]) -> dict[str, float]:
    values_np = np.array(list(values), dtype=np.float64)
    return {
        "mean": float(values_np.mean()),
        "std": float(values_np.std()),
        "min": float(values_np.min()),
        "max": float(values_np.max()),
    }


def _format_latex_summary(
    target_dataset: str,
    init_source: str,
    epoch_results: dict[int, list[dict[str, float]]],
) -> str:
    lines = [
        f"% {target_dataset.upper()} / {init_source}",
        "% epoch & accuracy",
    ]
    for epoch in sorted(epoch_results):
        summary = _summarize(row["accuracy"] for row in epoch_results[epoch])
        lines.append(
            f"{epoch} & ${summary['mean']:.2f}"
            f"{{\\scriptstyle\\pm {summary['std']:.2f}}}$ \\\\"
        )
    return "\n".join(lines) + "\n"


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _seed_everything(args.seed)
    device = torch.device(args.device) if args.device else _default_device()
    print(f"Device: {device}")
    print(f"Init source: {args.init_source}")
    print(f"Target dataset: {args.target_dataset}")
    protocol = _resolve_protocol_defaults(args)
    print(f"Protocol: {args.protocol}")

    initial_models = _load_initial_models(args, device)
    if (
        args.init_source == "generated"
        and protocol["source_bn_recalibration"]
        and initial_models
        and has_batchnorm(initial_models[0])
    ):
        cifar_bn_loader = _get_cifar10_bn_loader(
            data_root=args.data_root,
            batch_size=128,
            num_workers=args.num_workers,
            source_data_root=args.source_data_root,
        )
        initial_models = [
            recalibrate_bn_stats(model, calibration_loader=cifar_bn_loader, device=device)
            for model in initial_models
        ]

    train_loader, test_loader = _get_target_loaders(
        target_dataset=args.target_dataset,
        data_root=args.data_root,
        batch_size=protocol["batch_size"],
        num_workers=args.num_workers,
        stl10_label_space=protocol["stl10_label_space"],
        protocol=args.protocol,
    )
    requested_epochs = sorted(set(args.epochs))
    if 0 not in requested_epochs:
        requested_epochs.insert(0, 0)
    max_epoch = max(requested_epochs)

    epoch_results: dict[int, list[dict[str, float]]] = defaultdict(list)
    for model_idx, base_model in enumerate(initial_models):
        print(f"\n[{model_idx + 1}/{len(initial_models)}] Evaluating initializer")
        model = copy.deepcopy(base_model).to(device)
        if protocol["target_bn_recalibration"] and has_batchnorm(model):
            model = recalibrate_bn_stats(
                model,
                calibration_loader=train_loader,
                device=device,
            )

        loss, accuracy = _evaluate(model, test_loader, device)
        epoch_results[0].append({"loss": loss, "accuracy": accuracy})
        print(f"  epoch 0: acc={accuracy:.2f} loss={loss:.4f}")

        _freeze_feature_prefix(model, protocol["detach_ratio"])
        optimizer = _build_optimizer(
            model,
            optimizer_name=protocol["optimizer"],
            lr=protocol["lr"],
            weight_decay=protocol["weight_decay"],
            momentum=args.momentum,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
            if args.protocol == "deepweightflow" and max_epoch > 0
            else None
        )
        best_loss = loss
        best_accuracy = accuracy
        for epoch in range(1, max_epoch + 1):
            _train_one_epoch(model, train_loader, optimizer, device)
            if scheduler is not None:
                scheduler.step()
            if args.recalibrate_bn_after_epoch and has_batchnorm(model):
                model = recalibrate_bn_stats(
                    model,
                    calibration_loader=train_loader,
                    device=device,
                )
            if protocol["report_best"] or epoch in requested_epochs:
                loss, accuracy = _evaluate(model, test_loader, device)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_loss = loss
            if epoch in requested_epochs:
                if protocol["report_best"]:
                    loss, accuracy = best_loss, best_accuracy
                epoch_results[epoch].append({"loss": loss, "accuracy": accuracy})
                print(f"  epoch {epoch}: acc={accuracy:.2f} loss={loss:.4f}")

    summary = {
        str(epoch): {
            "accuracy": _summarize(row["accuracy"] for row in rows),
            "loss": _summarize(row["loss"] for row in rows),
            "per_model": rows,
        }
        for epoch, rows in sorted(epoch_results.items())
    }
    result = {
        "init_source": args.init_source,
        "target_dataset": args.target_dataset,
        "n_models": args.n_models,
        "epochs": requested_epochs,
        "protocol": args.protocol,
        "target_bn_recalibration": protocol["target_bn_recalibration"],
        "source_bn_recalibration": protocol["source_bn_recalibration"],
        "recalibrate_bn_after_epoch": args.recalibrate_bn_after_epoch,
        "stl10_label_space": protocol["stl10_label_space"] if args.target_dataset == "stl10" else None,
        "target_normalization": _target_normalization(args.target_dataset, args.protocol),
        "report_best": protocol["report_best"],
        "detach_ratio": protocol["detach_ratio"],
        "optimizer": {
            "name": protocol["optimizer"],
            "lr": protocol["lr"],
            "momentum": args.momentum if protocol["optimizer"] == "sgd" else None,
            "weight_decay": protocol["weight_decay"],
        },
        "result_dir": str(args.result_dir) if args.result_dir else None,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "summary": summary,
        "latex_summary": _format_latex_summary(
            args.target_dataset,
            args.init_source,
            epoch_results,
        ),
    }
    return result


def _default_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    base_dir = args.result_dir if args.result_dir is not None else Path("results")
    return (
        base_dir
        / "transfer_finetune"
        / f"{args.target_dataset}_{args.init_source}_n{args.n_models}.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-source",
        choices=["generated", "original", "random"],
        required=True,
        help="Initializer family to evaluate.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="DriftWeight result directory for generated initializers.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/CIFAR10_resnet18/checkpoint"),
        help="CIFAR-10 ResNet-18 seed checkpoint directory.",
    )
    parser.add_argument(
        "--target-dataset",
        choices=["stl10", "svhn"],
        required=True,
        help="Transfer target dataset.",
    )
    parser.add_argument(
        "--protocol",
        choices=["deepweightflow", "cifar_warmstart"],
        default="deepweightflow",
        help=(
            "Transfer protocol. deepweightflow matches the comparison code: "
            "target normalization, AdamW, partial feature freezing, native STL-10 labels, "
            "and best-so-far reporting."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[0, 1, 5],
        help="Fine-tuning epochs to report. Epoch 0 is always included.",
    )
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-end", type=int, default=None)
    parser.add_argument("--generated-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--optimizer",
        choices=["auto", "sgd", "adamw"],
        default="auto",
        help="Optimizer to use. auto selects AdamW for deepweightflow.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--detach-ratio",
        type=float,
        default=None,
        help=(
            "Fraction of early ResNet feature blocks to freeze. auto uses 0.4 for "
            "DeepWeightFlow non-random initializers and 0.0 for random."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/transfer"),
        help="Root directory for STL-10/SVHN downloads.",
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        default=None,
        help="Optional CIFAR-10 root for source BatchNorm recalibration.",
    )
    parser.add_argument(
        "--stl10-label-space",
        choices=["auto", "cifar10", "native"],
        default="auto",
        help=(
            "Map STL-10 labels into CIFAR-10 output order or keep native order. "
            "auto uses native for deepweightflow."
        ),
    )
    parser.add_argument(
        "--no-recalibrate-bn",
        dest="recalibrate_bn",
        action="store_false",
        help="Legacy switch: disables target BatchNorm recalibration for cifar_warmstart.",
    )
    parser.set_defaults(recalibrate_bn=True)
    parser.add_argument(
        "--target-bn-recalibration",
        action="store_true",
        help="Force epoch-0 BatchNorm recalibration on target training data.",
    )
    parser.add_argument(
        "--no-source-bn-recalibration",
        dest="source_bn_recalibration",
        action="store_false",
        help="Disable CIFAR-10 BatchNorm recalibration for generated weights.",
    )
    parser.set_defaults(source_bn_recalibration=True)
    parser.add_argument(
        "--report-best",
        action="store_true",
        help="Report best-so-far test accuracy at requested epochs.",
    )
    parser.add_argument(
        "--recalibrate-bn-after-epoch",
        action="store_true",
        help="Recompute BN stats after each fine-tuning epoch before evaluation.",
    )
    parser.add_argument(
        "--save-regenerated-models",
        action="store_true",
        help="When regenerating from flow_model.pt+pca.npz, save generated_models/.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = _run(args)
    output_path = _default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved transfer results to {output_path}")
    print("\nLaTeX summary:")
    print(result["latex_summary"])


if __name__ == "__main__":
    main()
