#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
command -v apptainer >/dev/null
mkdir -p .runtime/apptainer-cache
export APPTAINER_CACHEDIR="$project_dir/.runtime/apptainer-cache"
export APPTAINER_TMPDIR="${GARMI_RL_BUILD_TMPDIR:-/tmp}"
if [[ ! -d third_party/IsaacLab ]]; then
  git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
fi
test "$(git -C third_party/IsaacLab rev-parse HEAD)" = 37ddf626871758333d6ed89cf64ad702aef127d0
if [[ ! -d .runtime/container ]]; then
  # This host's /raid mount is absent from mountinfo, so build on /tmp first.
  build_dir=$(mktemp -d "$APPTAINER_TMPDIR/garmi-rl-build.XXXXXX")
  image=$(sed -n 's/^FROM //p' rl/Dockerfile)
  apptainer build --sandbox --fix-perms "$build_dir/container" "docker://$image"
  mv "$build_dir/container" .runtime/container
  rmdir "$build_dir"
fi
python3 scripts/prepare_apptainer.py .runtime/container
GARMI_RL_RUNTIME=apptainer bash scripts/container_exec.sh bash /workspace/rl/install.sh
printf 'apptainer\n' > .runtime/backend
echo 'Apptainer runtime ready. GPU selection: GARMI_RL_GPU (default 1).'
