#!/usr/bin/env bash
# Immutable during execution; endpoint physics and review only.
set -euo pipefail
batch="$1"
root="runs/validated-reference-bank-v5/batch-$batch"
bash scripts/rl.sh /workspace/graspgen/validate_rl.py \
  --config "/workspace/$root/source/task.yaml" \
  --out "/workspace/$root/pitcher/physics" --num_envs 80 --headless \
  --cases "/workspace/$root/pitcher/cases.json" --seed 2026101901 \
  --object-collision-mesh "/workspace/$root/pitcher/cooked-boundary.obj"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python \
  graspgen/pitcher_friction_validation.py review --batch "$batch"
