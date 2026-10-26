#!/usr/bin/env bash
# Train the full FedMIPT model on one task for every seed (Validation only).
#   bash scripts/train_main.sh task1_phone_sports
# Environment: SEEDS (default "1 2 42 99 2026"), DEVICE (default auto).
set -euo pipefail
threads="${CPU_THREADS:-4}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
task="${1:?usage: $0 task1_phone_sports}"
seeds="${SEEDS:-1 2 42 99 2026}"
for seed in $seeds; do
  python main.py train --config "configs/${task}.json" --seed "$seed" --device "${DEVICE:-auto}" --cpu_threads "$threads"
done
