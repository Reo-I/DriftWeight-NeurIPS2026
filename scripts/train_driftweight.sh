#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET="${1:-mnist}"
if [ "$#" -gt 0 ]; then
    shift
fi

case "$PRESET" in
    mnist)
        PRESET_ARGS=(--dataset mnist --arch mlp
            --checkpoint-dir data/MNIST/checkpoint
            --pca-components 4699 --mf-t1-ratio 0.375)
        ;;
    fashion_mnist)
        PRESET_ARGS=(--dataset fashion_mnist --arch mlp
            --checkpoint-dir data/FashionMNIST/checkpoint
            --pca-components 500 --mf-t1-ratio 0.2
            --drift-lambda 0.4 --drift-hardneg-k 256
            --drift-history-tail-exclude 0.0)
        ;;
    cifar10_resnet20)
        PRESET_ARGS=(--dataset cifar10 --arch resnet20
            --checkpoint-dir data/CIFAR10/checkpoint
            --pca-components 1999 --mf-t1-ratio 0.3
            --drift-target-ratio 0.15)
        ;;
    cifar10_resnet18)
        PRESET_ARGS=(--dataset cifar10 --arch resnet18
            --checkpoint-dir data/CIFAR10_resnet18/checkpoint
            --pca-components 500 --hidden-dim 4096 --time-embed-dim 128
            --mf-t1-ratio 0.375 --mf-onestep-lambda 0.5
            --drift-hardneg-k 256 --drift-history-tail-exclude 0.0)
        ;;
    -h|--help)
        printf 'Usage: bash scripts/train_driftweight.sh PRESET [training options]\n'
        printf 'Presets: mnist, fashion_mnist, cifar10_resnet20, cifar10_resnet18\n'
        exit 0
        ;;
    *)
        printf 'Unknown preset: %s\n' "$PRESET" >&2
        exit 2
        ;;
esac

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec python -m src.main.train_flow_model \
    --method meanflow \
    --n-iters 15000 --batch-size 16 --lr 1e-3 --weight-decay 1e-5 \
    --hidden-dim 1024 --time-embed-dim 64 --dropout 0.1 \
    --sigma 0.001 --source-std 0.01 --n-samples 100 --log-freq 100 \
    --use-pca --pca-fit-source all \
    --use-drift-loss --drift-lambda 0.25 --drift-target-ratio 0.1 \
    --drift-auto-scale ema --drift-ema-decay 0.99 --drift-scale-eps 1e-8 \
    --drift-temperatures 0.02,0.05,0.2 --drift-neg-max-total 2048 \
    --drift-same-ratio 0.25 --drift-history-tail-exclude 0.02 \
    --drift-a6-eps 1e-6 --drift-hardneg-k 512 \
    --mf-flow-ratio 0.25 --mf-r0-ratio 0.0 --mf-onestep-lambda 1.2 \
    --mf-time-dist uniform --mf-loss-gamma 0.25 --mf-jvp-api autograd \
    --eval-steps 1 --save-generated-models \
    --save-dir "results/${PRESET}" \
    "${PRESET_ARGS[@]}" "$@"
