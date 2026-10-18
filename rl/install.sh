#!/usr/bin/env bash
set -euo pipefail
# Run inside the container after mounting this project at /workspace.
/isaac-sim/python.sh -m pip install pip==25.3 'setuptools<81'
/isaac-sim/python.sh -m pip install --build-constraint /workspace/rl/build-constraints.txt \
  --constraint /workspace/rl/runtime-constraints.txt \
  -e /workspace/third_party/IsaacLab/source/isaaclab \
  -e /workspace/third_party/IsaacLab/source/isaaclab_assets \
  -e /workspace/third_party/IsaacLab/source/isaaclab_tasks \
  -e '/workspace/third_party/IsaacLab/source/isaaclab_rl[skrl]' \
  'skrl==1.4.3' 'torch==2.7.0' 'torchvision==0.22.0' 'python-fcl==0.7.0.10'
