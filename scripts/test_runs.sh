#!/usr/bin/env bash
# Evaluate fixed best checkpoints on Test, once per run.  Run this only after
# all configurations have been frozen on Validation.
#   bash scripts/test_runs.sh runs/task1_phone_sports_seed*
set -euo pipefail
threads="${CPU_THREADS:-4}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
[[ $# -gt 0 ]] || { echo "usage: $0 RUN_DIR..." >&2; exit 2; }
for run in "$@"; do
  python main.py test --run_dir "$run" --device "${DEVICE:-auto}" --cpu_threads "$threads"
done
