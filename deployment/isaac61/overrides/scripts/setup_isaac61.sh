#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if ! test -f /isaac-sim/VERSION || ! grep -q '^6\.1\.' /isaac-sim/VERSION; then
  echo 'This installer requires an existing Isaac Sim 6.1 at /isaac-sim.' >&2
  exit 1
fi
lab_dir="$project_dir/third_party/IsaacLab23Sim61"
if ! test -d "$lab_dir"; then
  git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$lab_dir"
fi
test "$(git -C "$lab_dir" rev-parse HEAD)" = 37ddf626871758333d6ed89cf64ad702aef127d0
patch_path="$project_dir/scripts/isaaclab23-sim61.patch"
if ! git -C "$lab_dir" apply --reverse --check "$patch_path" 2>/dev/null; then
  git -C "$lab_dir" apply --check "$patch_path"
  git -C "$lab_dir" apply "$patch_path"
fi
if ! test -x .venv61/bin/python; then
  uv venv --python /isaac-sim/kit/python/bin/python3 --seed .venv61
fi
uv pip install --python .venv61/bin/python \
  -r scripts/requirements-isaac61.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
uv pip install --python .venv61/bin/python --no-deps \
  --build-constraints rl/build-constraints.txt \
  -e "$lab_dir/source/isaaclab" \
  -e "$lab_dir/source/isaaclab_assets" \
  -e "$lab_dir/source/isaaclab_tasks" \
  -e "$lab_dir/source/isaaclab_rl"
uv pip check --python .venv61/bin/python
bash scripts/local_python61.sh -m pip check
