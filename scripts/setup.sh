#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
container_name="${GARMI_RL_CONTAINER:-garmi-mobile-grasp-rl}"
if [[ ! -d third_party/IsaacLab ]]; then
  git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
fi
test "$(git -C third_party/IsaacLab rev-parse HEAD)" = 37ddf626871758333d6ed89cf64ad702aef127d0
if ! docker container inspect "$container_name" >/dev/null 2>&1; then
  docker build -t garmi-mobile-grasp-rl:isaac51 -f rl/Dockerfile rl
  docker run -d --name "$container_name" --runtime nvidia --gpus all \
    --network host --ipc host --no-healthcheck --user 0:0 \
    --env NVIDIA_DRIVER_CAPABILITIES=all --volume "$project_dir:/workspace" \
    garmi-mobile-grasp-rl:isaac51
else
  mounted_root="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "$container_name")"
  test "$mounted_root" = "$project_dir" || { echo 'Container /workspace belongs to another project' >&2; exit 1; }
fi
docker start "$container_name" >/dev/null
docker exec "$container_name" bash /workspace/rl/install.sh
echo 'Runtime ready. Import validated assets before preparing a run; see README.md.'
