#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
unset PYTHONHOME PYTHONPATH ISAAC_PATH CARB_APP_PATH EXP_PATH CONDA_PREFIX LD_PRELOAD
unset LD_LIBRARY_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONEXE="$project_dir/.venv61/bin/python"
# Put the isolated environment before Isaac Sim's prebundled Python libraries.
export PYTHONPATH="$project_dir/.venv61/lib/python3.12/site-packages:$project_dir/integration:$project_dir/rl:$project_dir/scripts:$project_dir/cram-vrb-lab"
if [[ "${GARMI_ENABLE_ROS:-0}" == 1 ]]; then
  export ROS_DISTRO=jazzy
  export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.core/jazzy/lib
fi
exec /isaac-sim/python.sh --no-ros-env "$@"
