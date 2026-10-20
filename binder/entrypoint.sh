#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p results/cram-rl-web
python3 -u integration/web_server.py --external-jupyter > results/cram-rl-web/server.log 2>&1 &
# Binder supplies Jupyter's authentication and /user/<session>/ proxy prefix.
exec "$@"
