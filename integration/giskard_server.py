"""Launch the checked-out Giskard server with the container's installed ROS assets."""
from pathlib import Path
import runpy
from dataclasses import replace
from functools import partial
import sys

from cram_vrb_lab import paths

installed_ros = Path('/home/jovyan/cram-vrb-lab/ros2_ws')
if not (paths.ROS2_WS_DIR / 'src/iai_maps/iai_garmi_apartment/mjcf/scene-bodies.xml').exists():
    if not (installed_ros / 'src/iai_maps/iai_garmi_apartment/mjcf/scene-bodies.xml').exists():
        raise FileNotFoundError('The GARMI apartment ROS assets are missing')
    paths.ROS2_WS_DIR = installed_ros

from cram_vrb_lab.setups import SETUPS
from garmi_cram_rl.robot_interface import make_interface

key = ('garmi', 'garmi_apartment')
setup = SETUPS[key]
dual_arm = '--dual-arm' in sys.argv
if dual_arm:
    sys.argv.remove('--dual-arm')
SETUPS[key] = replace(setup, robot=replace(setup.robot,
    giskard_interface=partial(make_interface, dual_arm=dual_arm)))

runpy.run_path(str(paths.REPO_DIR / 'demos/giskard_server.py'), run_name='__main__')
