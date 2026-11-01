# DriftWeight

Code for **DriftWeight: Repulsive Drift in Mean-Flow Space for Neural Network
Weight Generation** (NeurIPS 2026), by Reo Iizuka, Hiroaki Shioya, and Naoto Yokoya.

DriftWeight learns a distribution of neural network weights using MeanFlow,
PCA fitted on final and intermediate training checkpoints, repulsive drift,
and a one-step consistency loss. Generated weights can be loaded into a target
classifier and evaluated without further training.

## Scope

This code-only release includes:

- Source-model training and checkpoint collection.
- PCA, MeanFlow, repulsive drift, and boundary consistency.
- Weight generation, classifier accuracy, and source-model error-set overlap
  (MaxIoU) evaluation.
- MNIST/MLP, Fashion-MNIST/MLP, and CIFAR-10/ResNet-20 or ResNet-18.
- Auxiliary scripts for source-model evaluation, transfer evaluation, and
  generator timing. The Flow Matching implementation is also retained.

The CIFAR-100/ViT-Base experiment is **not included** in this source snapshot.
Paper sources, PDFs, reviewer correspondence, experiment outputs, datasets,
and pretrained checkpoints are not bundled. This is not a claim that every
paper experiment is reproducible from this release.

## Installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Reo-I/DriftWeight-NeurIPS2026.git
cd DriftWeight-NeurIPS2026
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

The tests use small synthetic datasets on CPU and do not download benchmark
data. They check implementation behavior, not the reported paper accuracies.
The lockfile fixes dependency versions; full-scale experiments require
substantial compute, memory, and checkpoint storage. CUDA is recommended for
paper-scale runs. Training otherwise selects MPS or CPU automatically.
Both training entry points accept `--device cpu|mps|cuda|auto` for explicit control.

## Local Validation

A real MNIST run completed source training, PCA fitting, 15,000 generator
iterations, saved-weight evaluation, MaxIoU, and generator checkpoint reload.
It used 16 source models and 751 PCA components, not the full 100-model preset.
Mean accuracies for 100 generated models were **88.92% / 95.74% / 96.25%**
at 1 / 5 / 100 steps. This confirms the workflow runs, **not paper-level
accuracy or diversity reproduction**; the one-step result needs further
investigation. See [VALIDATION.md](VALIDATION.md) for conditions, commands,
checks, and remaining release requirements.

## MNIST Workflow

Run commands from the repository root. The first command downloads MNIST through
torchvision and trains 100 source classifiers, saving both final and intermediate
checkpoints. Choose unused checkpoint and output directories for each run.
The training commands reject directories containing earlier run artifacts
instead of overwriting weights or mixing samples from different runs.

```bash
uv run --locked python -m src.main.generate_training_data \
  --dataset mnist --arch mlp --num-seeds 100 \
  --epochs 5 --batch-size 64 --lr 0.001 --optimizer adam \
  --hidden-dim 32 --save-every 100 \
  --checkpoint-dir data/MNIST/checkpoint

uv run --locked bash scripts/train_driftweight.sh mnist \
  --save-dir results/mnist

uv run --locked python scripts/compute_maxiou.py \
  --result-dir results/mnist \
  --checkpoint-dir data/MNIST/checkpoint --device cpu
```

The training command also generates and evaluates one-step weights.
MaxIoU measures overlap with source classifiers'
error sets; it is not a complete measure of diversity among generated models
or of ensemble benefit. CPU or CUDA is recommended for MaxIoU; the evaluator
avoids MPS by default because small classifiers can yield different argmaxes.

For multi-step accuracy evaluation, append `--eval-steps 1 5 100` to the
training command. Each step count has its own directory, such as
`generated_models/steps_1/` or `generated_models/steps_100/`, so samples are
not overwritten by a different step count. `evaluation_results.json` records
the matching path. MaxIoU uses `--steps 1` by default; specify another step
count to evaluate those saved weights. Legacy flat directories are accepted
only when their recorded last-evaluated step count matches the request.
Each MaxIoU result is saved to `maxiou_steps_<N>.json` by default.

## Other Supported Experiments

Collect source checkpoints before running the corresponding generator preset:

```bash
# Fashion-MNIST / MLP
uv run --locked python -m src.main.generate_training_data \
  --dataset fashion_mnist --arch mlp --num-seeds 100 \
  --epochs 25 --batch-size 128 --lr 0.001 --optimizer adamw \
  --hidden-dim 128 --save-every 100 \
  --checkpoint-dir data/FashionMNIST/checkpoint
uv run --locked bash scripts/train_driftweight.sh fashion_mnist \
  --save-dir results/fashion_mnist

# CIFAR-10 / ResNet-20
uv run --locked python -m src.main.generate_training_data \
  --dataset cifar10 --arch resnet20 --num-seeds 100 \
  --epochs 5 --batch-size 128 --lr 0.001 --optimizer adam \
  --save-every 100 --checkpoint-dir data/CIFAR10/checkpoint
uv run --locked bash scripts/train_driftweight.sh cifar10_resnet20 \
  --save-dir results/cifar10_resnet20

# CIFAR-10 / ResNet-18
uv run --locked python -m src.main.generate_training_data \
  --dataset cifar10 --arch resnet18 --num-seeds 100 \
  --epochs 100 --batch-size 128 --lr 0.1 --optimizer sgd \
  --save-every 1000 --checkpoint-dir data/CIFAR10_resnet18/checkpoint
uv run --locked bash scripts/train_driftweight.sh cifar10_resnet18 \
  --save-dir results/cifar10_resnet18
```

The generator presets encode the reported main-run settings. Shared settings
include 15,000 iterations, batch size 16, learning rate 0.001, drift temperatures
0.02/0.05/0.2, EMA scaling, PCA fitted on final plus history checkpoints, and
100 generated models. Dataset-specific settings are explicit in
`scripts/train_driftweight.sh`; extra CLI arguments override preset values.
These recipes have not been rerun at paper scale during release preparation.

Keep at least as many final checkpoints as the generator batch size. Drift
training also needs intermediate checkpoints: collect at least three
`checkpoint_iter*.pt` files per seed so history remains after final-checkpoint
and history-tail filtering. Do not mix datasets, architectures,
or MLP widths in a checkpoint directory. ResNet evaluation recalibrates BatchNorm
statistics using the evaluation inputs, matching this implementation's protocol.
The original checkpoint pool is not distributed, so retraining can change
numerical results. Source initialization, data-loader randomness, software, and
hardware also affect results.

## Outputs and Utilities

A generator run writes `flow_model.pt`, `pca.npz`, `config.json`,
`evaluation_results.json`, and `generated_models/` under `--save-dir`.
`RESULTS_DIR`, when set, overrides `--save-dir`; unset it to use the paths above.
Additional logs are written under `log/`. These outputs are ignored by Git.

```bash
uv run --locked python -m src.main.generate_training_data --help
uv run --locked python -m src.main.train_flow_model --help
uv run --locked python scripts/compute_maxiou.py --help
uv run --locked python scripts/evaluate_original_final.py --help
uv run --locked python scripts/evaluate_transfer_finetune.py --help
uv run --locked python scripts/benchmark_generation_speed.py --help
```

Transfer evaluation supports the ResNet-18 workflow; consult its CLI for the
target dataset and protocol. Timing covers the generator and optionally PCA
decoding, not the full source-training or generator-training pipeline.
Only load checkpoints from trusted sources.

## Layout

```text
src/           Models, data loaders, PCA, losses, training, and evaluation
scripts/       Dataset presets and focused evaluation utilities
tests/         CPU smoke and regression tests
VALIDATION.md  Real MNIST validation results and limitations
pyproject.toml Minimal runtime dependencies
uv.lock        Locked dependency versions
CITATION.cff   Citation metadata
```

The existing `src` Python package name is retained for import compatibility.
The release copy restores source generation to seeds `0 .. num_seeds - 1`,
removing a development-only resume offset. The research repository and its
Git history are not part of this repository.

## Citation

```bibtex
@inproceedings{iizuka2026driftweight,
  title = {DriftWeight: Repulsive Drift in Mean-Flow Space for Neural Network Weight Generation},
  author = {Iizuka, Reo and Shioya, Hiroaki and Yokoya, Naoto},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```

## Release Status

This is a public, code-only release. Source-model training, generator training,
and evaluation scripts are provided; pretrained weights are not distributed.
The experiment coverage and validation limitations are described above and in
[VALIDATION.md](VALIDATION.md). A redistribution license has not yet been
specified.
