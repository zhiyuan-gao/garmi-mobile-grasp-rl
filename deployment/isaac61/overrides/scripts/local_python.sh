#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default to the validated Isaac Sim 6.1 compatibility environment.
# The original 5.1 entry point remains available as local_python51.sh.
exec bash "$project_dir/scripts/local_python61.sh" "$@"
