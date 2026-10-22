#!/usr/bin/env bash
# Run the standalone demo against the frozen mug release and an existing runtime.
set -euo pipefail
demo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$demo_root/../.." && pwd)"
workspace="${GARMI_DRAWER_WORKSPACE:-$repo_root/.runtime/mug-ppo6500}"
runtime="${GARMI_DRAWER_RUNTIME:-$repo_root/.runtime/container}"
third_party="${GARMI_DRAWER_THIRD_PARTY:-$repo_root/third_party}"
run_cache="$demo_root/.runtime"
for required in "$runtime" "$runtime/home" "$workspace/rl" "$third_party"; do
  if [[ ! -d "$required" ]]; then
    echo "Missing required directory: $required. See demos/mug_drawer/README.md." >&2
    exit 1
  fi
done
if [[ $# -eq 0 ]]; then
  echo 'Usage: bash demos/mug_drawer/scripts/sim.sh scripts/record_mug_drawer.py --headless --out runs/example' >&2
  exit 2
fi
mkdir -p "$run_cache"/{home,cache,kit-cache,kit-data,kit-logs}
exec apptainer exec --nv --cleanenv \
  --home "$run_cache/home:/home/rl" \
  --bind "$runtime/home:/home" --bind "$run_cache/home:/home/rl" \
  --bind "$workspace:/workspace" --bind "$demo_root:/demo" \
  --bind "$third_party:/workspace/third_party:ro" \
  --bind "$run_cache/cache:/demo-cache" \
  --bind "$run_cache/kit-cache:/isaac-sim/kit/cache" \
  --bind "$run_cache/kit-data:/isaac-sim/kit/data" \
  --bind "$run_cache/kit-logs:/isaac-sim/kit/logs" \
  --pwd /demo --env PYTHONPATH=/workspace/rl:/demo \
  --env XDG_CACHE_HOME=/demo-cache \
  --env OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 \
  --env ACCEPT_EULA=YES,PRIVACY_CONSENT=YES,OMNI_KIT_ACCEPT_EULA=YES,PYTHONUNBUFFERED=1 \
  --env "CUDA_VISIBLE_DEVICES=${GARMI_DRAWER_GPU:-0}" \
  "$runtime" /isaac-sim/python.sh "$@"
