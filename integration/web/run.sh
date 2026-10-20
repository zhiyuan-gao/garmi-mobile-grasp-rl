#!/usr/bin/env bash
set -euo pipefail
# Run in the foreground: Ctrl-C stops this page's services and active demo.
exec docker exec -it fame-rl-dev python3 -u /workspace/integration/web_server.py "$@"
