#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
exec bash "$project_dir/scripts/container_exec.sh" /isaac-sim/python.sh "$@"
