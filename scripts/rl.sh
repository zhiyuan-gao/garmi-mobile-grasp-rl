#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
docker exec -e PYTHONPATH=/workspace/rl "${GARMI_RL_CONTAINER:-garmi-mobile-grasp-rl}" /isaac-sim/python.sh "$@"
