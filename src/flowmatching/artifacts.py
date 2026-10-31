"""Locate generated weights without confusing sampling step counts."""

import json
from pathlib import Path


def generated_models_dir(run_dir: Path, n_steps: int) -> Path:
    if n_steps < 1:
        raise ValueError("Sampling step count must be positive.")
    return Path(run_dir) / "generated_models" / f"steps_{n_steps}"


def resolve_generated_models_dir(run_dir: Path, n_steps: int) -> Path:
    """Prefer step-specific weights; validate the last-step legacy layout."""
    run_dir = Path(run_dir)
    selected = generated_models_dir(run_dir, n_steps)
    if selected.is_dir():
        return selected

    legacy = run_dir / "generated_models"
    if not any(legacy.glob("generated_model_*.pt")):
        return selected
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise ValueError("Legacy generated weights need config.json to identify their step count.")
    config = json.loads(config_path.read_text())
    steps = config.get("eval_steps", [])
    if not steps or any(not isinstance(step, int) or step < 1 for step in steps):
        raise ValueError("Cannot determine the sampling step count of legacy generated weights.")
    saved_steps = max(steps)
    if saved_steps != n_steps:
        raise ValueError(
            f"Legacy generated weights are from {saved_steps} steps, not {n_steps}. "
            "Select the matching step count or regenerate the requested weights."
        )
    return legacy
