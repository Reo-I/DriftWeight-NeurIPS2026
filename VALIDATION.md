# Local release validation

Date: 2026-10-31. This is a functional release check, not a reproduction of
the paper's reported accuracy, diversity, or GPU latency.

## Environment and protocol

- Apple M3 Max, 14 CPU cores, 36 GiB unified memory.
- Python 3.12.13, PyTorch 2.10.0, torchvision 0.25.0, NumPy 2.4.2,
  scikit-learn 1.8.0, using `uv.lock`.
- Real MNIST: all 60,000 training images and all 10,000 test images.
- Sixteen source MLPs, architecture 784-32-32-10, seeds 0 through 15.
  Five epochs each, Adam at 0.001, batch size 64, intermediate checkpoints
  every 100 iterations. Source training used MPS.
- PCA fitted on 16 final and 736 retained intermediate checkpoints:
  752 fitting samples, 751 components. The full MNIST preset instead uses
  100 source models and 4,699 components.
- Generator training used the MNIST preset for 15,000 iterations, batch size
  16, hidden dimension 1,024, and time embedding dimension 64. Loss settings
  were unchanged. Only checkpoint/output paths, PCA rank, device, logging
  interval, and evaluation step counts were overridden.
- Generator training and evaluation used CPU with four numerical-library
  threads. Each step count used 100 newly generated classifiers; classifiers
  were not fine-tuned. These are not paired samples across step counts.

## Results

Accuracy is the mean and population standard deviation across classifiers,
in percent. Source results evaluate the saved `final.pt` files, not each
source model's best intermediate epoch.

| Models | Count | Mean accuracy | Standard deviation | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source, final checkpoints | 16 | 96.3575 | 0.1855 | 96.02 | 96.69 |
| Generated, 1 step | 100 | 88.9218 | 2.2022 | 76.08 | 92.09 |
| Generated, 5 steps | 100 | 95.7429 | 1.0740 | 85.13 | 96.22 |
| Generated, 100 steps | 100 | 96.2526 | 0.1286 | 95.99 | 96.75 |

The one-step result is substantially below the source population. This run
does **not** establish paper-level one-step accuracy. The reduced source
population, PCA rank, and different hardware are differences from the full
recipe, not a demonstrated explanation for the gap.

Independent MaxIoU evaluation reloaded the saved weights for 1 and 100 steps:

| Error-set comparison | Mean MaxIoU | Standard deviation |
| --- | ---: | ---: |
| Generated 1-step vs. sources | 0.2758 | 0.0443 |
| Generated 100-step vs. sources | 0.9227 | 0.0237 |
| Sources vs. other sources, leave-one-out | 0.4530 | 0.0116 |

The low one-step overlap is not sufficient evidence of useful diversity,
given its lower accuracy. Conversely, 100-step samples have high overlap
with source-model errors. Neither result establishes the paper's diversity
claim in this reduced setting.

Observed wall time was 268.89 seconds for source training and 262.86 seconds
for PCA, generator training, and all three accuracy evaluations. Independent
MaxIoU runs took 22.70 and 22.08 seconds. These are local run timings, not
controlled performance benchmarks.

## Commands

Run from the repository root with fresh output directories. MNIST is
downloaded automatically. `RESULTS_DIR` must not override the specified path.
Source training below explicitly selects the MPS device used by this run;
select a supported device on other machines.

```bash
uv sync --locked
unset RESULTS_DIR

uv run --locked python -m src.main.generate_training_data \
  --dataset mnist --arch mlp --num-seeds 16 --epochs 5 \
  --batch-size 64 --lr 0.001 --optimizer adam --hidden-dim 32 \
  --save-every 100 --device mps \
  --checkpoint-dir data/validation_mnist16/checkpoint

env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  uv run --locked bash scripts/train_driftweight.sh mnist --device cpu \
  --checkpoint-dir data/validation_mnist16/checkpoint \
  --pca-components 751 --eval-steps 1 5 100 --log-freq 1000 \
  --save-dir results/validation_mnist16

env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  uv run --locked python scripts/evaluate_original_final.py \
  --dataset mnist --arch mlp --device cpu --batch-size 256 \
  --checkpoint-dir data/validation_mnist16/checkpoint

for steps in 1 100; do
  env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
    uv run --locked python scripts/compute_maxiou.py \
    --result-dir results/validation_mnist16 \
    --checkpoint-dir data/validation_mnist16/checkpoint \
    --device cpu --steps "$steps"
done

env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
  uv run --locked python scripts/benchmark_generation_speed.py \
  --run-dir results/validation_mnist16 --steps 1 5 100 --n-samples 100 \
  --warmup-runs 2 --timed-runs 5 --include-inverse-pca --device cpu \
  --output results/validation_mnist16/reload_benchmark_cpu.json

uv run --locked python -m unittest discover -s tests -v
```

## Checks and fixes

- All 16 regression/smoke tests pass, including a synthetic end-to-end run,
  preset argument checks, PCA round-trip, drift loss, and weight flattening.
- Python compilation, shell syntax checking, and `uv pip check` pass.
  The 46 candidate Git files contain no paper/presentation artifacts,
  datasets, weights, logs, local user paths, or matches for the checked
  credential patterns. This scan is not a guarantee against every secret format.
- All 300 real generated weights reload as finite 26,506-element tensors.
  The saved generator state is finite and reloads for 1/5/100-step generation
  plus inverse PCA. No invalid-loss or error messages were found in the
  real generator training log.
- Each sampling step count now has a separate weights directory and MaxIoU
  output. Previously, later step counts overwrote earlier generated weights.
  Transfer evaluation now selects the requested step count. Legacy flat
  weight directories are accepted only when their recorded step count matches.
- Per-model saving now clones tensor views so each file contains only that
  model's storage. CPU output previously serialized the entire 100-model
  backing tensor into each file (about 10 MB instead of 0.1 MB per MNIST model).
  The local artifacts were compacted with exact tensor-value equality checked;
  this does not alter the reported accuracies. Regression tests check storage size.
- Training rejects existing run artifacts instead of mixing or overwriting
  checkpoints. A too-large generator batch fails early with a useful message.
- Explicit CPU/MPS/CUDA selection is available in both training CLIs.
  MPS timing now synchronizes before measuring elapsed time.
- MaxIoU now recalibrates ResNet-20 BatchNorm just as the main evaluator does.
  This branch selection is regression-tested, not validated on real CIFAR-10.
- Removed stale project names, private experiment examples, unused commented
  paths, fixed-rank PCA descriptions, and ambiguous old "paper best" comments.
  The learning equations and preset loss hyperparameters were not changed.

Raw data, checkpoints, generated weights, and logs remain local and Git-ignored.
The numerical JSON artifacts are under `results/validation_mnist16/`; the
source evaluation is in `data/validation_mnist16/checkpoint/training_stats.json`.

## Remaining validation and licensing work

1. Investigate one-step accuracy using the full 100-source MNIST recipe before
   claiming numerical reproduction. Multiple generator/source runs would be
   needed to characterize variability.
2. Real Fashion-MNIST, CIFAR-10, transfer, CUDA, and large streaming-PCA runs
   were not executed in this check. CIFAR-100/ViT is absent from this snapshot.
3. A redistribution license has not yet been selected. License terms and
   third-party attribution remain to be confirmed by the authors.

On 2026-11-01, the author approved making the code publicly available.
This does not change the validation limitations above.

The original research repository, manuscript, reviewer records, and paper
results were not modified by this release validation.
