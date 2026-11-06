"""Collision-aware Giskard probes in a disposable CRAM world."""
from copy import deepcopy
from dataclasses import dataclass, field
import math
import time

from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.robot_plans.motions.navigation import MoveMotion
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


@dataclass
class NavigationProbeMotion(MoveMotion):
    """Solve a path: the stock simulated MoveMotion uses SetOdometry instead."""
    @property
    def _motion_chart(self):
        return CartesianPose(root_link=self.world.root,tip_link=self.robot.root,goal_pose=self.target)


@dataclass
class NavigationProbeTrace(PlanCallback):
    positions: list = field(default_factory=list)

    def on_motion_tick(self,statechart):
        matrix=self.plan.robot.root.global_pose.to_np()
        self.positions.append([float(matrix[0,3]),float(matrix[1,3])])


def copied_context(demo):
    demo.sync.pull(demo.world)
    world=deepcopy(demo.world)
    robot=world.get_semantic_annotation_by_id(demo.robot.id)
    return demo.context.for_world(world,robot)


def probe_navigation(demo,goal):
    started=time.monotonic()
    context=copied_context(demo)
    height=float(context.robot.root.global_pose.to_np()[2,3])
    pose=HomogeneousTransformationMatrix.from_xyz_rpy(goal[0],goal[1],height,yaw=goal[2],
                                                     reference_frame=context.world.root).to_pose()
    try:
        node=execute_single(NavigationProbeMotion(target=pose,keep_joint_states=True),context=context)
        trace=NavigationProbeTrace(plan=node.plan)
        node.plan.node_callbacks.append(trace)
        with simulated_robot(collision_avoidance=True):
            node.perform()
        matrix=context.robot.root.global_pose.to_np()
        actual=[float(matrix[0,3]),float(matrix[1,3]),math.atan2(matrix[1,0],matrix[0,0])]
        error=math.dist(actual[:2],goal[:2])
        angle=abs(math.atan2(math.sin(actual[2]-goal[2]),math.cos(actual[2]-goal[2])))
        # Check the achieved pose explicitly as well as chart completion.
        passed=error<=.03 and angle<=.01
        demo.record('station_probe',goal=goal,actual=actual,passed=passed,
                    seconds=time.monotonic()-started,planning_world_copy=True,
                    solver_steps=len(trace.positions),
                    path_length_m=sum(math.dist(a,b) for a,b in zip(trace.positions,trace.positions[1:])))
        return passed
    except Exception as error:
        demo.record('station_probe',goal=goal,passed=False,error=repr(error),
                    seconds=time.monotonic()-started,planning_world_copy=True)
        return False


def probe_preparation(demo,open_gripper=True):
    context=copied_context(demo)
    motions=[]
    if open_gripper:
        motions.append(MoveGripperMotion(motion=GripperState.OPEN,gripper=Arms.RIGHT))
    motions.append(MoveJointsMotion(names=[f'right_fr3_joint{i}' for i in range(1,8)],
        positions=[0.,-math.pi/4,0.,-3*math.pi/4,0.,math.pi/2,math.pi/4]))
    try:
        with simulated_robot(collision_avoidance=True):
            sequential(motions,context=context).perform()
        demo.record('recovery_probe',passed=True,planning_world_copy=True)
        return True
    except Exception as error:
        demo.record('recovery_probe',passed=False,error=repr(error),planning_world_copy=True)
        return False
