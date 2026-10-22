"""Launch the checked-out Giskard server with the container's installed ROS assets."""
from pathlib import Path
import argparse
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

def main():
    # Match the frozen 41c1032 robot package. The host's demo launcher can
    # require newer scene-joint APIs that this static RL scene does not have.
    from cram_vrb_lab.control.giskard_server import start_localization_stand_in
    from cram_vrb_lab.control.rate import control_hz
    from cram_vrb_lab.setups import add_setup_arguments, get_setup, spawn_pose_from_args
    from giskardpy.middleware.ros2 import rospy
    from giskardpy.middleware.ros2.giskard import Giskard
    from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
    from giskardpy.qp.qp_controller_config import QPControllerConfig

    parser = argparse.ArgumentParser(description=__doc__)
    add_setup_arguments(parser)
    parser.add_argument('--control-hz', type=float, default=None)
    args = parser.parse_args()
    selected = get_setup(args.robot, args.scene)
    environment = selected.scene.environment() if selected.scene.environment else None
    rate = args.control_hz if args.control_hz is not None else control_hz()
    rospy.init_node('giskard')
    localization = start_localization_stand_in(selected.robot.base_link_height)
    try:
        giskard = Giskard(
            world_config=selected.robot.giskard_world(environment, spawn_pose_from_args(args)),
            robot_interface_config=selected.robot.giskard_interface(),
            server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
            qp_controller_config=QPControllerConfig(target_frequency=rate, prediction_horizon=15))
        giskard.live()
    finally:
        localization.terminate()


if __name__ == '__main__':
    main()
