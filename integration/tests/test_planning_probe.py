from types import SimpleNamespace

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.world import World
from semantic_digital_twin.spatial_types.spatial_types import Pose
from garmi_cram_rl.planning_probe import NavigationProbeMotion


def test_navigation_probe_solves_motion_instead_of_overwriting_odometry():
    world=World()
    context=Context(world=world,robot=SimpleNamespace(root=world.root))
    motion=NavigationProbeMotion(Pose.from_xyz_rpy(1.,0.,0.,reference_frame=world.root))
    execute_single(motion,context=context)
    with simulated_robot:
        chart=motion.motion_chart
    assert isinstance(chart,CartesianPose)
