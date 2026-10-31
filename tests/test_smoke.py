"""CPU checks with synthetic inputs; no benchmark data or GPU required."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.compute_maxiou import _masks_to_iou_matrix
from src.cli import FlowTrainingArgs, parse_flow_args
from src.config import GenerateConfig
from src.flowmatching.drifting import compute_V
from src.flowmatching.pca import WeightSpacePCA
from src.flowmatching.weight_space import WeightSpaceObjectStateDict
from src.main import generate_training_data, train_flow_model
from src.models import MLP


ROOT = Path(__file__).resolve().parents[1]


class ReleaseSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_weight_round_trip(self):
        original = MLP(hidden_dim=4, seed=0)
        flat = WeightSpaceObjectStateDict.from_model(original).flatten()
        restored = MLP(hidden_dim=4, seed=1)
        WeightSpaceObjectStateDict.from_flat(flat, restored).apply_to_model(restored)
        for first, second in zip(original.parameters(), restored.parameters()):
            torch.testing.assert_close(first, second)

    def test_pca_round_trip_and_save(self):
        weights = torch.randn(8, 3)
        pca = WeightSpacePCA(n_components=3).fit(weights)
        projected = pca.transform(weights)
        torch.testing.assert_close(pca.inverse_transform(projected), weights)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "pca.npz")
            pca.save(path)
            loaded = WeightSpacePCA.load(path)
            torch.testing.assert_close(loaded.transform(weights), projected)

    def test_drift_field_is_finite(self):
        current = torch.randn(4, 3)
        positives = torch.randn(5, 3)
        negatives = torch.cat([current, torch.randn(6, 3)])
        field = compute_V(current, positives, negatives, 0.05, True)
        self.assertEqual(field.shape, current.shape)
        self.assertTrue(torch.isfinite(field).all())
        with self.assertRaises(ValueError):
            compute_V(current, positives, negatives, 0.0)

    def test_error_set_iou(self):
        generated = np.array([[True, False, True], [False, True, False]])
        sources = np.array([[True, False, True], [True, True, False]])
        np.testing.assert_allclose(
            _masks_to_iou_matrix(generated, sources),
            np.array([[1.0, 1.0 / 3.0], [0.0, 0.5]]),
        )

    def test_synthetic_generation_and_driftweight_pipeline(self):
        data = TensorDataset(torch.randn(12, 1, 28, 28), torch.arange(12) % 10)
        loader = DataLoader(data, batch_size=4, shuffle=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoint"
            source_config = GenerateConfig(
                num_seeds=2, num_epochs=1, batch_size=4,
                checkpoint_dir=checkpoint_dir, save_every_n_iterations=1,
                device="cpu",
            )
            with (
                patch.object(generate_training_data, "parse_generate_args", return_value=source_config),
                patch.object(generate_training_data, "get_mnist_loaders", return_value=(loader, loader)),
            ):
                generate_training_data.main()

            self.assertEqual(
                sorted(path.parent.name for path in checkpoint_dir.glob("*/final.pt")),
                ["seed0000", "seed0001"],
            )
            stats = json.loads((checkpoint_dir / "training_stats.json").read_text())
            self.assertEqual(stats["n_models"], 2)
            self.assertEqual(len(stats["accuracies"]), 2)
            run_dir = root / "run"
            training_config = FlowTrainingArgs(
                method="meanflow", checkpoint_dir=checkpoint_dir, save_dir=run_dir,
                device="cpu", n_iters=2, batch_size=2, hidden_dim=16,
                time_embed_dim=4, dropout=0.0, n_samples=2, eval_steps=[1, 2],
                pca_components=3, pca_fit_source="all", use_drift_loss=True,
                drift_neg_max_total=8, drift_hardneg_k=4,
                drift_history_tail_exclude=0.0, mf_flow_ratio=0.25,
                mf_t1_ratio=0.375, mf_onestep_lambda=1.2,
                mf_loss_gamma=0.25, log_freq=1,
            )
            with (
                patch.object(train_flow_model, "parse_flow_args", return_value=training_config),
                patch.object(train_flow_model, "get_mnist_loaders", return_value=(loader, loader)),
            ):
                train_flow_model.main()
            for filename in ("flow_model.pt", "pca.npz", "config.json", "evaluation_results.json"):
                self.assertTrue((run_dir / filename).is_file(), filename)
            generated = sorted((run_dir / "generated_models").glob("generated_model_*.pt"))
            self.assertEqual(len(generated), 2)
            for path in generated:
                self.assertTrue(torch.isfinite(torch.load(path, weights_only=True)).all())
            checkpoint = torch.load(run_dir / "flow_model.pt", weights_only=True)
            self.assertTrue(math.isfinite(checkpoint["best_loss"]))
            config = json.loads((run_dir / "config.json").read_text())
            self.assertTrue(config["use_drift_loss"])
            self.assertEqual(config["pca_fit_source"], "all")

    def test_preset_arguments_and_overrides(self):
        expected = {
            "mnist": ("mnist", "mlp", 4699, 1024, 512, 0.375, 0.25, 0.1, 1.2),
            "fashion_mnist": ("fashion_mnist", "mlp", 500, 1024, 256, 0.2, 0.4, 0.1, 1.2),
            "cifar10_resnet20": ("cifar10", "resnet20", 1999, 1024, 512, 0.3, 0.25, 0.15, 1.2),
            "cifar10_resnet18": ("cifar10", "resnet18", 500, 4096, 256, 0.375, 0.25, 0.1, 0.5),
        }
        with tempfile.TemporaryDirectory() as directory:
            python_stub = Path(directory) / "python"
            python_stub.write_text(
                f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            python_stub.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{directory}{os.pathsep}{env.get('PATH', '')}"
            for preset, values in expected.items():
                with self.subTest(preset=preset):
                    result = subprocess.run(
                        ["bash", "scripts/train_driftweight.sh", preset, "--n-iters", "7"],
                        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
                    )
                    argv = json.loads(result.stdout)
                    self.assertEqual(argv[:2], ["-m", "src.main.train_flow_model"])
                    with (
                        patch.object(sys, "argv", ["train_flow_model", *argv[2:]]),
                        patch("src.cli.flow_args.get_available_device", return_value="cpu"),
                    ):
                        args = parse_flow_args()
                    self.assertEqual(
                        (args.dataset, args.arch, args.pca_components, args.hidden_dim,
                         args.drift_hardneg_k, args.mf_t1_ratio, args.drift_lambda,
                         args.drift_target_ratio, args.mf_onestep_lambda),
                        values,
                    )
                    self.assertEqual(args.n_iters, 7)
                    self.assertEqual(args.eval_steps, [1])
                    self.assertTrue(args.use_drift_loss)
                    self.assertEqual(args.pca_fit_source, "all")

    def test_command_line_entry_points(self):
        commands = [
            [sys.executable, "-m", "src.main.generate_training_data", "--help"],
            [sys.executable, "-m", "src.main.train_flow_model", "--help"],
        ]
        commands.extend(
            [sys.executable, str(path), "--help"]
            for path in sorted((ROOT / "scripts").glob("*.py"))
        )
        commands.extend(
            ["bash", "scripts/train_driftweight.sh", preset, "--help"]
            for preset in ("mnist", "fashion_mnist", "cifar10_resnet20", "cifar10_resnet18")
        )
        env = os.environ.copy()
        env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                        text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
