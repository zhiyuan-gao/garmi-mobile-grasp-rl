#!/usr/bin/env bash
set -euo pipefail
# A separate container and output directory; existing demos and services are untouched.
DEMO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$DEMO/../.." && pwd)
: "${GARMI_BOWL_WORKSPACE:?Set to the original validated bowl/FAME_RL workspace}"
: "${GARMI_MUG_WORKSPACE:?Set to the validated mug update-6500 workspace}"
: "${GARMI_ISAACLAB:?Set to the existing IsaacLab checkout}"
MODE=${1:-run}
if [[ "$MODE" != prepare && "$MODE" != run && "$MODE" != preview ]]; then
  echo 'Usage: docker.sh prepare|preview|run' >&2; exit 2
fi
NAME="garmi-household-${MODE}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEMO/assets/cache"
COMMON=(--runtime=nvidia --gpus all --ipc=host -e ACCEPT_EULA=YES -e PRIVACY_CONSENT=YES -e OMNI_KIT_ACCEPT_EULA=YES
  -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONDONTWRITEBYTECODE=1
  --mount "type=bind,src=$GARMI_ISAACLAB,dst=/workspace/third_party/IsaacLab,readonly"
  --mount "type=bind,src=$DEMO/assets/cache,dst=/isaac-sim/kit/cache" \
  --mount "type=bind,src=$REPO,dst=/source,readonly" --mount "type=bind,src=$DEMO,dst=/demo")
IMAGE=${GARMI_DEMO_IMAGE:-garmi-rl-fork:validation}
if [[ "$MODE" == prepare ]]; then
  docker run --name "$NAME" "${COMMON[@]}" --mount "type=bind,src=$GARMI_MUG_WORKSPACE,dst=/workspace,readonly" \
    --entrypoint bash "$IMAGE" -lc 'export PYTHONPATH=/workspace/rl:/workspace/third_party/IsaacLab/source/isaaclab:/workspace/third_party/IsaacLab/source/isaaclab_rl:/home/jovyan/cram-vrb-lab; /isaac-sim/python.sh /source/demos/mug_drawer/scripts/prepare_mug_drawer.py && /isaac-sim/python.sh /source/demos/mug_drawer/scripts/prepare_mug_drawer_apartment.py'
else
  # Stage small code files in this demo's private workspace; all large inputs
  # are mounted read-only. No mountpoint is created in either source workspace.
  mkdir -p "$DEMO/assets/workspace/integration/assets" "$DEMO/assets/workspace/rl/assets"
  cp -a "$REPO/integration/." "$DEMO/assets/workspace/integration/"
  cp -a "$REPO/runtime" "$DEMO/assets/workspace/"
  cp -a "$REPO/rl/garmi_rl" "$DEMO/assets/workspace/rl/"
  cp -a "$REPO/scripts" "$DEMO/assets/workspace/"
  EXTRA=(); [[ "$MODE" != preview ]] || EXTRA+=(--preview-only)
  docker run --name "$NAME" "${COMMON[@]}" \
    --mount "type=bind,src=$DEMO/assets/workspace,dst=/workspace" \
    --mount "type=bind,src=$GARMI_BOWL_WORKSPACE/results,dst=/workspace/results,readonly" \
    --mount "type=bind,src=$GARMI_BOWL_WORKSPACE/rl/assets,dst=/workspace/rl/assets,readonly" \
    --mount "type=bind,src=$GARMI_BOWL_WORKSPACE/cram-vrb-lab,dst=/workspace/cram-vrb-lab,readonly" \
    --mount "type=bind,src=$GARMI_MUG_WORKSPACE,dst=/mug,readonly" \
    --mount "type=bind,src=$GARMI_BOWL_WORKSPACE/integration/assets,dst=/workspace/integration/assets,readonly" \
    --entrypoint python3 "$IMAGE" /demo/run.py --out "/demo/runs/$NAME" "${EXTRA[@]}"
fi
