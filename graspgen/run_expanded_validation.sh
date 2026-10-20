#!/usr/bin/env bash
# Only physical diagnostics and review. Deliberately no expert/training commands.
set -euo pipefail
name="$1"
batch="$2"
test -f "$batch/$name/preparation.json"
python=.runtime/graspgen-venv/bin/python
bash scripts/rl.sh /workspace/graspgen/validate_rl.py \
  --config "/workspace/$batch/$name/task.yaml" \
  --out "/workspace/$batch/$name/physics" --num_envs 80 --headless \
  --object-collision-mesh "/workspace/$batch/$name/cooked-boundary.obj"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$python" graspgen/expand_reference_validation.py repeat \
  --object "$name" --out "$batch"
case_count=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["case_count"])' \
  "$batch/repeat-$name/$name/inputs.json")
if [ "$case_count" -eq 0 ]; then
  "$python" graspgen/expand_reference_validation.py review --object "$name" --out "$batch"
  exit 0
fi
bash scripts/rl.sh /workspace/graspgen/validate_rl.py \
  --config "/workspace/$batch/$name/task.yaml" \
  --out "/workspace/$batch/repeat-$name/$name/physics" --num_envs 80 --headless \
  --cases "/workspace/$batch/repeat-$name/$name/cases.json" --seed 2026101901 \
  --object-collision-mesh "/workspace/$batch/repeat-$name/$name/cooked-boundary.obj"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$python" graspgen/expand_reference_validation.py review \
  --object "$name" --out "$batch"
