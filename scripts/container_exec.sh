#!/usr/bin/env bash
# Keep identical container paths with either Docker or an unprivileged Apptainer runtime.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="${GARMI_RL_RUNTIME:-$(cat "$project_dir/.runtime/backend" 2>/dev/null || echo docker)}"
case "$backend" in
  docker)
    exec docker exec -e PYTHONPATH=/workspace/rl -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
      -e "GARMI_RL_PROJECT_ROOT=$project_dir" \
      "${GARMI_RL_CONTAINER:-garmi-mobile-grasp-rl}" "$@"
    ;;
  apptainer)
    runtime="$project_dir/.runtime/container"
    test -d "$runtime" || { echo 'Run bash scripts/setup_apptainer.sh first' >&2; exit 1; }
    mkdir -p "$project_dir/.runtime/home" "$project_dir/.runtime/cache"
    exec apptainer exec --nv --writable --cleanenv \
      --home "$project_dir/.runtime/home:/home/rl" \
      --bind "$runtime/home:/home" --bind "$project_dir/.runtime/home:/home/rl" \
      --bind "$project_dir:/workspace" --pwd /workspace \
      --env XDG_CACHE_HOME=/workspace/.runtime/cache \
      --env PYTHONPATH=/workspace/rl,OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1 \
      --env "GARMI_RL_PROJECT_ROOT=$project_dir" \
      --env ACCEPT_EULA=YES,PRIVACY_CONSENT=YES,OMNI_KIT_ACCEPT_EULA=YES,PYTHONUNBUFFERED=1 \
      --env "CUDA_VISIBLE_DEVICES=${GARMI_RL_GPU:-1}" "$runtime" "$@"
    ;;
  *) echo "Unknown GARMI_RL_RUNTIME: $backend" >&2; exit 2 ;;
esac
