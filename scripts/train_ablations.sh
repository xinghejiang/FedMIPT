#!/usr/bin/env bash
# Ablation variants of the paper (RQ2).  Each variant changes exactly one
# group of switches relative to configs/TASK.json.
#   bash scripts/train_ablations.sh task1_phone_sports
set -euo pipefail
threads="${CPU_THREADS:-4}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
task="${1:?usage: $0 TASK}"
seeds="${SEEDS:-1 2 42 99 2026}"
# tag:override[,override...]
variants=(
  "no_itemcl:item_cl_weight=0"
  "no_usercl:user_cl_weight=0"
  "no_cl:item_cl_weight=0,user_cl_weight=0"
  "no_text:use_text=false"
  "no_image:use_image=false"
  # Without any content there is no semantic space, hence no transfer.
  "no_content:use_text=false,use_image=false,cross_transfer=false"
  # Strict: no interest packages and no projector aggregation.
  "no_cross:cross_transfer=false"
)
for variant in "${variants[@]}"; do
  tag="${variant%%:*}"
  overrides=()
  IFS=',' read -r -a pairs <<< "${variant#*:}"
  for kv in "${pairs[@]}"; do overrides+=(--set "$kv"); done
  for seed in $seeds; do
    python main.py train --config "configs/${task}.json" --seed "$seed" --tag "$tag" \
      --device "${DEVICE:-auto}" --cpu_threads "$threads" "${overrides[@]}"
  done
done
