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
from src.cli import FlowTrainingArgs, parse_flow_args, parse_generate_args
from src.config import GenerateConfig
from src.flowmatching.artifacts import generated_models_dir, resolve_generated_models_dir
from src.flowmatching.drifting import compute_V
from src.flowmatching.pca import WeightSpacePCA
from src.flowmatching.weight_dataset import create_flow_dataloaders
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
            evaluation = json.loads((run_dir / "evaluation_results.json").read_text())
            for steps in (1, 2):
                generated_dir = generated_models_dir(run_dir, steps)
                generated = sorted(generated_dir.glob("generated_model_*.pt"))
                self.assertEqual(len(generated), 2)
                self.assertEqual(
                    run_dir / evaluation[str(steps)]["generated_models_dir"], generated_dir
                )
                for path in generated:
                    weight = torch.load(path, weights_only=True)
                    self.assertTrue(torch.isfinite(weight).all())
                    self.assertEqual(
                        weight.untyped_storage().nbytes(), weight.numel() * weight.element_size()
                    )
            checkpoint = torch.load(run_dir / "flow_model.pt", weights_only=True)
            self.assertTrue(math.isfinite(checkpoint["best_loss"]))
            config = json.loads((run_dir / "config.json").read_text())
            self.assertTrue(config["use_drift_loss"])
            self.assertEqual(config["pca_fit_source"], "all")

    def test_cpu_override_does_not_auto_select_mps(self):
        for parser, module in (
            (parse_flow_args, "src.cli.flow_args"),
            (parse_generate_args, "src.cli.generate_args"),
        ):
            with self.subTest(parser=module), patch.object(sys, "argv", ["cli", "--device", "cpu"]):
                with patch(f"{module}.get_available_device", return_value="mps") as auto_device:
                    self.assertEqual(parser().device, "cpu")
                    auto_device.assert_not_called()

    def test_existing_runs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "checkpoint"
            final = source_dir / "seed0000" / "final.pt"
            final.parent.mkdir(parents=True)
            final.write_bytes(b"existing source checkpoint")
            source_config = GenerateConfig(checkpoint_dir=source_dir, device="cpu")
            with patch.object(generate_training_data, "parse_generate_args", return_value=source_config):
                with self.assertRaisesRegex(FileExistsError, "new --checkpoint-dir"):
                    generate_training_data.main()
            self.assertEqual(final.read_bytes(), b"existing source checkpoint")

            output_dir = root / "run"
            output_dir.mkdir()
            model = output_dir / "flow_model.pt"
            model.write_bytes(b"existing generator")
            args = FlowTrainingArgs(save_dir=output_dir, device="cpu")
            with patch.object(train_flow_model, "parse_flow_args", return_value=args):
                with self.assertRaisesRegex(FileExistsError, "new --save-dir"):
                    train_flow_model.main()
            self.assertEqual(model.read_bytes(), b"existing generator")

    def test_step_specific_weights_are_not_overwritten(self):
        model = MLP(seed=0)
        flat = WeightSpaceObjectStateDict.from_model(model).flatten().unsqueeze(0)
        loader = DataLoader(TensorDataset(torch.zeros(1, 1, 28, 28), torch.zeros(1, dtype=torch.long)))
        with tempfile.TemporaryDirectory() as directory:
            args = FlowTrainingArgs(save_dir=Path(directory), device="cpu", n_samples=1)
            train_flow_model.evaluate_generated_weights(flat, args, MLP, loader, n_steps=1)
            first_file = generated_models_dir(args.save_dir, 1) / "generated_model_0000.pt"
            first_content = first_file.read_bytes()
            train_flow_model.evaluate_generated_weights(flat + 0.1, args, MLP, loader, n_steps=5)
            self.assertEqual(first_content, first_file.read_bytes())
            second = torch.load(generated_models_dir(args.save_dir, 5) / first_file.name, weights_only=True)
            torch.testing.assert_close(second, (flat + 0.1)[0])

    def test_legacy_weights_require_matching_step_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            legacy = run_dir / "generated_models"
            legacy.mkdir()
            (legacy / "generated_model_0000.pt").touch()
            (run_dir / "config.json").write_text(json.dumps({"eval_steps": [1, 5, 100]}))
            with self.assertRaisesRegex(ValueError, "100 steps, not 1"):
                resolve_generated_models_dir(run_dir, 1)
            self.assertEqual(resolve_generated_models_dir(run_dir, 100), legacy)
            specific = generated_models_dir(run_dir, 1)
            specific.mkdir()
            self.assertEqual(resolve_generated_models_dir(run_dir, 1), specific)

    def test_transfer_loader_selects_requested_step_count(self):
        from scripts.evaluate_transfer_finetune import _generated_files

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for steps in (1, 5):
                output_dir = generated_models_dir(root, steps)
                output_dir.mkdir(parents=True)
                (output_dir / "generated_model_0000.pt").touch()
            self.assertEqual(
                _generated_files(root, 5),
                [generated_models_dir(root, 5) / "generated_model_0000.pt"],
            )

    def test_resnet_maxiou_recalibrates_batchnorm(self):
        from scripts import compute_maxiou

        train, test = object(), object()
        with patch.object(compute_maxiou, "get_cifar10_loaders", return_value=(train, test)):
            for arch in ("resnet20", "resnet18"):
                with self.subTest(arch=arch):
                    loader, calibration = compute_maxiou._get_test_loader("cifar10", arch)
                    self.assertIs(loader, test)
                    self.assertIs(calibration, test)

    def test_leave_one_out_requires_two_models(self):
        from scripts.compute_maxiou import _loo_maxiou

        with self.assertRaisesRegex(ValueError, "at least two"):
            _loo_maxiou(np.array([[True, False]]))

    def test_batch_size_cannot_exceed_checkpoint_count(self):
        weights = [WeightSpaceObjectStateDict.from_model(MLP(hidden_dim=4, seed=seed)) for seed in range(2)]
        with self.assertRaisesRegex(ValueError, "exceeds the number of final checkpoints"):
            create_flow_dataloaders(weights, [0, 1], batch_size=3, device="cpu", use_pca=False)

    def test_mps_timing_synchronizes(self):
        from scripts.benchmark_generation_speed import _synchronize

        with patch("torch.mps.synchronize") as synchronize:
            _synchronize(torch.device("mps"))
            synchronize.assert_called_once()

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
