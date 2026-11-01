#!/usr/bin/env python3
"""Benchmark weight-generation latency from saved DriftWeight run artifacts.

Example:
    uv run scripts/benchmark_generation_speed.py \
        --run-dir results/driftweight_mnist \
        --run-dir results/deepweightflow_mnist \
        --steps 1 5 100 \
        --n-samples 100 \
        --warmup-runs 5 \
        --timed-runs 100 \
        --require-gpu-substring H100 \
        --output results/generation_speed_h100.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.flowmatching.flow_model import WeightSpaceFlowModel
from src.flowmatching.meanflow_model import WeightSpaceMeanFlowModel
from src.flowmatching.pca import WeightSpacePCA


Method = Literal["flowmatching", "meanflow"]
Solver = Literal["euler", "rk4"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _load_checkpoint(run_dir: Path, device: torch.device) -> dict[str, Any]:
    model_path = run_dir / "flow_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    return torch.load(model_path, map_location=device)


def _infer_method(config: dict[str, Any], checkpoint: dict[str, Any]) -> Method:
    method = config.get("method") or checkpoint.get("method")
    if method in {"flowmatching", "meanflow"}:
        return method

    state_dict = checkpoint.get("model_state_dict", {})
    has_meanflow_time = any(str(key).startswith("r_embedder.") for key in state_dict)
    return "meanflow" if has_meanflow_time else "flowmatching"


def _input_dim(config: dict[str, Any], checkpoint: dict[str, Any]) -> int:
    value = checkpoint.get("input_dim") or config.get("actual_dim")
    if value is None:
        raise ValueError("Cannot infer input dimension from checkpoint/config.")
    return int(value)


def _build_model(
    method: Method,
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    input_dim: int,
) -> torch.nn.Module:
    hidden_dim = int(config.get("hidden_dim") or state_dict["net.0.weight"].shape[0])
    if method == "meanflow":
        time_key = "t_embedder.0.weight"
    else:
        time_key = "time_embed.0.weight"
    time_embed_dim = int(config.get("time_embed_dim") or state_dict[time_key].shape[0])
    dropout = float(config.get("dropout", 0.1))

    if method == "meanflow":
        return WeightSpaceMeanFlowModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
        )
    return WeightSpaceFlowModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        dropout=dropout,
    )


def _load_model(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], Method, int]:
    config_path = run_dir / "config.json"
    config = _load_json(config_path) if config_path.exists() else {}
    checkpoint = _load_checkpoint(run_dir, device=device)
    method = _infer_method(config=config, checkpoint=checkpoint)
    input_dim = _input_dim(config=config, checkpoint=checkpoint)
    state_dict = checkpoint["model_state_dict"]

    model = _build_model(
        method=method,
        config=config,
        state_dict=state_dict,
        input_dim=input_dim,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config, method, input_dim


def _meanflow_generate(
    model: torch.nn.Module,
    *,
    n_samples: int,
    input_dim: int,
    source_std: float,
    n_steps: int,
    device: torch.device,
) -> torch.Tensor:
    z = torch.randn(n_samples, input_dim, device=device) * source_std
    t_vals = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    for step_idx in range(n_steps):
        t = torch.full((n_samples, 1), t_vals[step_idx], device=device)
        r = torch.full((n_samples, 1), t_vals[step_idx + 1], device=device)
        u = model(z, t, r)
        z = z - (t - r) * u
    return z


def _flowmatching_generate(
    model: torch.nn.Module,
    *,
    n_samples: int,
    input_dim: int,
    source_std: float,
    n_steps: int,
    solver: Solver,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn(n_samples, input_dim, device=device) * source_std
    dt = 1.0 / float(n_steps)

    for step_idx in range(n_steps):
        t_value = step_idx * dt
        t = torch.full((n_samples, 1), t_value, device=device)
        if solver == "euler":
            v = model(x, t)
            x = x + dt * v
            continue

        half_dt = 0.5 * dt
        k1 = model(x, t)
        k2 = model(x + half_dt * k1, t + half_dt)
        k3 = model(x + half_dt * k2, t + half_dt)
        k4 = model(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _stats(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean_ms": mean,
        "std_ms": math.sqrt(variance),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _nfe(method: Method, solver: Solver, n_steps: int) -> int:
    if method == "meanflow":
        return n_steps
    return n_steps * (4 if solver == "rk4" else 1)


def _make_generate_fn(
    model: torch.nn.Module,
    *,
    method: Method,
    solver: Solver,
    n_samples: int,
    input_dim: int,
    source_std: float,
    n_steps: int,
    device: torch.device,
    pca: WeightSpacePCA | None,
    include_inverse_pca: bool,
) -> Callable[[], torch.Tensor]:
    def generate_once() -> torch.Tensor:
        if method == "meanflow":
            generated = _meanflow_generate(
                model,
                n_samples=n_samples,
                input_dim=input_dim,
                source_std=source_std,
                n_steps=n_steps,
                device=device,
            )
        else:
            generated = _flowmatching_generate(
                model,
                n_samples=n_samples,
                input_dim=input_dim,
                source_std=source_std,
                n_steps=n_steps,
                solver=solver,
                device=device,
            )

        if include_inverse_pca:
            if pca is None:
                raise RuntimeError("--include-inverse-pca was set, but pca.npz is missing.")
            generated = pca.inverse_transform(generated, device=str(device))
        return generated

    return generate_once


def _benchmark(
    generate_once: Callable[[], torch.Tensor],
    *,
    n_samples: int,
    warmup_runs: int,
    timed_runs: int,
    device: torch.device,
) -> dict[str, Any]:
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = generate_once()
        _synchronize(device)

        per_run_ms: list[float] = []
        for _ in range(timed_runs):
            _synchronize(device)
            start = time.perf_counter()
            _ = generate_once()
            _synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            per_run_ms.append(elapsed_ms)

    per_model_ms = [value / n_samples for value in per_run_ms]
    return {
        "per_run_ms": _stats(per_run_ms),
        "per_model_ms": _stats(per_model_ms),
        "timed_runs": timed_runs,
        "warmup_runs": warmup_runs,
        "n_samples": n_samples,
    }


def _hardware_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": props.name,
                "total_memory_gb": props.total_memory / (1024**3),
                "cuda_version": torch.version.cuda,
            }
        )
    return info


def _default_label(method: Method) -> str:
    return "DriftWeight" if method == "meanflow" else "DeepWeightFlow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark generation latency from saved DriftWeight run directories."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=Path,
        help="Saved run directory containing flow_model.pt and optionally config.json/pca.npz. Repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label for each --run-dir. Repeat in the same order.",
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 5, 100])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--timed-runs", type=int, default=100)
    parser.add_argument(
        "--flowmatching-solver",
        choices=["euler", "rk4"],
        default=None,
        help="Override solver for flowmatching runs. Defaults to config integration_method or rk4.",
    )
    parser.add_argument(
        "--include-inverse-pca",
        action="store_true",
        help="Include PCA inverse transform in the timed region.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Benchmark device.",
    )
    parser.add_argument(
        "--require-gpu-substring",
        default=None,
        help="Fail if the CUDA device name does not include this substring, e.g. H100.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/generation_speed_benchmark.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.label is not None and len(args.label) != len(args.run_dir):
        raise ValueError("--label must be provided once per --run-dir when used.")
    if any(step <= 0 for step in args.steps):
        raise ValueError("--steps values must be positive.")
    if args.n_samples <= 0 or args.warmup_runs < 0 or args.timed_runs <= 0:
        raise ValueError("Invalid run counts or sample count.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    hardware = _hardware_info(device)
    gpu_name = str(hardware.get("gpu_name", ""))
    if args.require_gpu_substring and args.require_gpu_substring not in gpu_name:
        raise RuntimeError(
            f"Expected GPU name to contain {args.require_gpu_substring!r}, got {gpu_name!r}."
        )

    results: list[dict[str, Any]] = []
    for run_idx, run_dir in enumerate(args.run_dir):
        run_dir = run_dir.resolve()
        model, config, method, input_dim = _load_model(run_dir, device=device)
        source_std = float(config.get("source_std", 0.01))
        pca_path = run_dir / "pca.npz"
        pca = WeightSpacePCA.load(str(pca_path)) if pca_path.exists() else None
        label = args.label[run_idx] if args.label is not None else _default_label(method)
        dataset = config.get("dataset", "unknown")
        arch = config.get("arch", "unknown")

        solver: Solver = "euler"
        if method == "flowmatching":
            solver = args.flowmatching_solver or config.get("integration_method", "rk4")
            if solver not in {"euler", "rk4"}:
                raise ValueError(f"Unsupported flowmatching solver: {solver}")

        for n_steps in args.steps:
            generate_once = _make_generate_fn(
                model,
                method=method,
                solver=solver,
                n_samples=args.n_samples,
                input_dim=input_dim,
                source_std=source_std,
                n_steps=n_steps,
                device=device,
                pca=pca,
                include_inverse_pca=args.include_inverse_pca,
            )
            bench = _benchmark(
                generate_once,
                n_samples=args.n_samples,
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
                device=device,
            )
            row = {
                "run_dir": str(run_dir),
                "label": label,
                "dataset": dataset,
                "arch": arch,
                "method": method,
                "solver": solver if method == "flowmatching" else "euler",
                "steps": n_steps,
                "nfe": _nfe(method, solver, n_steps),
                "include_inverse_pca": args.include_inverse_pca,
                "source_std": source_std,
                "input_dim": input_dim,
                **bench,
            }
            results.append(row)

            per_model = row["per_model_ms"]
            print(
                f"{label:16s} {dataset}/{arch:10s} "
                f"steps={n_steps:3d} nfe={row['nfe']:4d} "
                f"time/model={per_model['mean_ms']:.3f} +/- {per_model['std_ms']:.3f} ms"
            )

    payload = {
        "hardware": hardware,
        "settings": {
            "steps": args.steps,
            "n_samples": args.n_samples,
            "warmup_runs": args.warmup_runs,
            "timed_runs": args.timed_runs,
            "include_inverse_pca": args.include_inverse_pca,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved benchmark JSON to {args.output}")


if __name__ == "__main__":
    main()
