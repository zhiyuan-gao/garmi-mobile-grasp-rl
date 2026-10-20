#!/usr/bin/env bash
# Physical tests of new conditions only; never expert collection or training.
set -euo pipefail
batch="$1"
bash scripts/rl.sh /workspace/graspgen/validate_rl.py \
  --config "/workspace/$batch/pitcher/task.yaml" \
  --out "/workspace/$batch/repeat-pitcher-extra/pitcher/physics" --num_envs 80 --headless \
  --cases "/workspace/$batch/repeat-pitcher-extra/pitcher/cases.json" --seed 2026101901 \
  --object-collision-mesh "/workspace/$batch/repeat-pitcher-extra/pitcher/cooked-boundary.obj"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .runtime/graspgen-venv/bin/python - "$batch" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,'graspgen')
import review_grasp_repeatability as review
batch=Path(sys.argv[1]).resolve()
review.OUT=batch/'repeat-pitcher-extra';review.SOURCES={'pitcher':batch/'pitcher'}
review.review('pitcher')
PY
