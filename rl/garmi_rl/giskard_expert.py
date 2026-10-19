"""Giskard approach controller for the frozen GARMI RL kitchen.

Run in the existing CRAM Python environment. The world uses the training URDF
and exported PhysX convex parts. No navigation, attachment, or grasp execution
is performed here; commands are executed and judged by the RL environment.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import copy
import pickle
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose, CartesianTask
from giskardpy.motion_statechart.goals.collision_avoidance import SelfCollisionAvoidance, ExternalCollisionAvoidance
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import Pose, HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection, OmniDrive, Connection6DoF
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions, AvoidSelfCollisions, AllowCollisionBetweenGroups, AvoidCollisionBetweenGroups,
)
from garmi_rl.expert_actions import PARK, ARM_TARGET_LOWER, ARM_TARGET_UPPER


@dataclass(eq=False)
class TrainingRobot(AbstractRobot):
    """Semantic boundary of the already frozen training robot."""

    @classmethod
    def get_ros_file_path(cls):
        return '/workspace/rl/assets/garmi_training.urdf'

    @classmethod
    def _get_root_body_name(cls):
        return 'base_link'

    def _setup_collision_rules(self):
        # Rules are installed after matching the training collision geometry.
        pass


@dataclass
class ApproachExpert:
    """Reusable Giskard QP, updated from measured states for each query."""

    root: Path = field(default_factory=lambda: Path('/workspace'))
    control_dt: float = .05

    def __post_init__(self):
        started = time.perf_counter()
        source = self.root / 'results/rl-preflight/bowl-kitchen-full'
        tree = ET.parse(self.root / 'rl/assets/garmi_training.urdf')
        # Geometry comes from the simulator export, not a second approximation.
        for link in tree.getroot().findall('link'):
            for tag in ('visual', 'collision'):
                for element in list(link.findall(tag)):
                    link.remove(element)
        kinematics = copy.deepcopy(tree.getroot())
        # The training articulation locks this mounting joint at zero.
        kinematics.find("joint[@name='right_fr3_joint0']").set('type', 'fixed')
        robot_world = URDFParser(urdf=ET.tostring(kinematics, encoding='unicode')).parse()
        self.world = World()
        with self.world.modify_world():
            origin = Body(name=PrefixedName('task'))
            odom = Body(name=PrefixedName('odom'))
            self.world.add_connection(FixedConnection(parent=origin, child=odom,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(z=.0259)))
            self.drive = OmniDrive.create_with_dofs(world=self.world, parent=odom,
                child=robot_world.root, translation_velocity_limits=.15, rotation_velocity_limits=.4)
            self.world.merge_world(robot_world, self.drive)
            for dof in self.world.degrees_of_freedom:
                dof.has_hardware_interface = False
            self.drive.has_hardware_interface = True
            self.arm = [self.world.get_connection_by_name(f'right_fr3_joint{i}') for i in range(1, 8)]
            for joint in self.arm:
                joint.has_hardware_interface = True
                joint.raw_dof._overwrite_dof_limits(DerivativeMap(velocity=-1.), DerivativeMap(velocity=1.))
            for dof, acceleration in zip((self.drive.x_velocity, self.drive.y_velocity, self.drive.yaw), (.3, .3, .8)):
                dof._overwrite_dof_limits(DerivativeMap(acceleration=-acceleration), DerivativeMap(acceleration=acceleration))
            self.fingers = [self.world.get_connection_by_name(f'right_fr3_finger_joint{i}') for i in (1, 2)]
            for joint in self.fingers:
                joint.has_hardware_interface = False
                self.world.state[joint.raw_dof.id].position = .04
                joint.raw_dof._overwrite_dof_limits(DerivativeMap(velocity=0.), DerivativeMap(velocity=0.))
            pieces = pickle.load((source / 'cooked-model.pkl').open('rb'))
            bowl_shapes = []
            robot_shapes = {}
            for record, mesh in pieces:
                shape = Mesh.from_trimesh(mesh)
                if record['parent'] == 'Object':
                    bowl_shapes.append(shape)
                else:
                    robot_shapes.setdefault(record['body'], []).append(shape)
            for name, shapes in robot_shapes.items():
                body = self.world.get_body_by_name(name)
                body.collision = ShapeCollection(shapes, reference_frame=body)
            self.robot = TrainingRobot(root=self.world.get_body_by_name('base_link'))
            self.world.add_semantic_annotation_recursively(self.robot)
            self.bowl = Body(name=PrefixedName('target_bowl'), collision=ShapeCollection(bowl_shapes))
            self.bowl_connection = Connection6DoF.create_with_dofs(world=self.world, parent=origin, child=self.bowl)
            self.world.add_connection(self.bowl_connection)
            self.external_bodies = [self.bowl]
            for index, (_, mesh) in enumerate(pickle.load((source / 'scene_meshes.pkl').open('rb'))):
                body = Body(name=PrefixedName(f'furniture_{index}'), collision=ShapeCollection([Mesh.from_trimesh(mesh)]))
                self.world.add_connection(FixedConnection(parent=origin, child=body))
                self.external_bodies.append(body)
            self.target = Body(name=PrefixedName('approach_target'))
            self.target_connection = Connection6DoF.create_with_dofs(world=self.world, parent=origin, child=self.target)
            self.world.add_connection(self.target_connection)
            self._install_collision_rules(tree.getroot(), robot_shapes)
        # A task-specific override belongs in the temporary rules. Default-rule
        # changes serialize the scene; passing the entire kitchen there is costly.
        hand_bodies = [self.world.get_body_by_name(name) for name in
            ('right_fr3_link7', 'right_fr3_leftfinger', 'right_fr3_rightfinger')]
        self.world.collision_manager.add_temporary_rule(AvoidCollisionBetweenGroups(
            body_group_a=hand_bodies, body_group_b=self.external_bodies,
            buffer_zone_distance=.001, violated_distance=0.))
        self.hand = self.world.get_body_by_name('right_fr3_hand')
        bank = json.loads((self.root / 'results/rl-preflight/bowl-kitchen-40refs/selected-40.json').read_text())
        self.references = {r['id']: r for r in bank['objects'][0]['references']}
        self.context = MotionStatechartContext(world=self.world,
            qp_controller_config=QPControllerConfig(target_frequency=1/self.control_dt, prediction_horizon=7, verbose=False))
        self.executor = Executor(self.context)
        self.chart = MotionStatechart()
        self.goal = CartesianPose(root_link=origin, tip_link=self.hand,
            goal_pose=Pose.from_xyz_rpy(reference_frame=self.target),
            translation_threshold=.008, orientation_threshold=.08)
        self.chart.add_nodes([self.goal, SelfCollisionAvoidance(robot=self.robot), ExternalCollisionAvoidance(robot=self.robot)])
        self.compiled = False
        self.initialization_seconds = time.perf_counter()-started

    def _install_collision_rules(self, urdf, shapes):
        """Match training's fixed-body/adjacent exclusions and finger contact."""
        manager = self.world.collision_manager
        manager.add_default_rule(AvoidSelfCollisions(robot=self.robot, buffer_zone_distance=.03, violated_distance=0.))
        manager.add_default_rule(AvoidExternalCollisions(robot=self.robot, buffer_zone_distance=.05, violated_distance=0.))
        # A generic 5 cm stand-off blocks the validated bowl grasps. Keep exact
        # collision checking for the hand/fingers, using the training contact
        # offset (1 mm) as the approach buffer; only finger/bowl contact is free.
        groups = {'base_link': 'base_link'}
        pending = list(urdf.findall('joint'))
        adjacent = set()
        while pending:
            ready = []
            for joint in pending:
                parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                if parent not in groups:
                    continue
                movable = joint.get('type') != 'fixed'
                groups[child] = child if movable else groups[parent]
                if movable:
                    adjacent.add(frozenset((groups[parent], groups[child])))
                ready.append(joint)
            if not ready:
                raise ValueError('Disconnected training URDF')
            pending = [j for j in pending if j not in ready]
        adjacent.add(frozenset(('right_fr3_leftfinger', 'right_fr3_rightfinger')))
        # Reuse GARMI's existing Giskard SRDF exclusions where both bodies are
        # represented separately. Never expand an exclusion to a merged torso.
        srdf = self.root/'cram-vrb-lab/cognitive_robot_abstract_machine/semantic_digital_twin/resources/collision_configs/garmi.srdf'
        for element in ET.parse(srdf).getroot().findall('disable_self_collision'):
            names = (element.get('link1'), element.get('link2'))
            if all(name in shapes for name in names):
                adjacent.add(frozenset(names))
        for pair in adjacent:
            names = list(pair)
            if len(names) == 2 and all(name in shapes for name in names):
                manager.add_ignore_collision_rule(AllowCollisionBetweenGroups(
                    body_group_a=[self.world.get_body_by_name(names[0])],
                    body_group_b=[self.world.get_body_by_name(names[1])]))
        manager.add_ignore_collision_rule(AllowCollisionBetweenGroups(
            body_group_a=[self.world.get_body_by_name(name) for name in ('right_fr3_leftfinger', 'right_fr3_rightfinger')],
            body_group_b=[self.bowl]))

    def set_pose(self, connection, pose):
        """Write an Isaac wxyz pose into passive planning degrees of freedom."""
        for dof, value in zip((connection.x, connection.y, connection.z,
                               connection.qw, connection.qx, connection.qy, connection.qz), pose[:7]):
            self.world.state[dof.id].position = float(value)

    def sync(self, request):
        snapshot = request['snapshot']
        root = np.asarray(snapshot['robot_state'])
        rotation = Rotation.from_quat(root[[4, 5, 6, 3]])
        self.world.state.velocities[:] = 0
        self.world.state.accelerations[:] = 0
        self.world.state.jerks[:] = 0
        for dof, value in zip((self.drive.x, self.drive.y, self.drive.yaw), [*root[:2], rotation.as_euler('xyz')[2]]):
            self.world.state[dof.id].position = float(value)
        local_velocity = rotation.inv().apply(root[7:10])
        for dof, value in zip((self.drive.x_velocity, self.drive.y_velocity, self.drive.yaw), [*local_velocity[:2], root[12]]):
            self.world.state[dof.id].velocity = float(value)
        self.world.state[self.drive.x_velocity.id].position = 0
        self.world.state[self.drive.y_velocity.id].position = 0
        for joint, position, velocity in zip(self.arm, snapshot['arm_position'], snapshot['arm_velocity']):
            joint.position = float(position)
            joint.velocity = float(velocity)
        for index, joint in enumerate(self.fingers):
            joint.position = .04 + snapshot['observation'][7+index]
        self.set_pose(self.bowl_connection, snapshot['object_state'])
        self.world.notify_state_change(publish_changes=False)
        ref = self.references[request['reference']]
        obj = np.asarray(snapshot['object_state'])
        object_rotation = Rotation.from_quat(obj[[4, 5, 6, 3]])
        goal_rotation = object_rotation * Rotation.from_quat(np.array(ref['quaternion'])[[1, 2, 3, 0]])
        goal_position = obj[:3] + object_rotation.apply(ref['position'])
        hand = self.hand.global_transform.to_np()
        offset = goal_rotation.inv().apply(hand[:3, 3]-goal_position)
        angle = (goal_rotation.inv()*Rotation.from_matrix(hand[:3, :3])).magnitude()
        # Same two-pose sequence as CRAM ReachAction / the mobile grasp demo.
        # The caller carries the stage across queries and across CPU workers.
        # This avoids switching back to the pre-pose while inserting the fingers.
        stage = request.get('approach_stage', 'approach_grasp')
        pre_position = goal_position-goal_rotation.apply([0., 0., .08])
        pre_distance = float(np.linalg.norm(hand[:3, 3]-pre_position))
        if stage == 'pregrasp' and pre_distance < .015 and angle < .15:
            stage = 'approach_grasp'
        if stage not in ('pregrasp', 'approach_grasp'):
            raise ValueError(f'Unknown approach stage {stage}')
        command_position = pre_position if stage == 'pregrasp' else goal_position
        command_quaternion = goal_rotation.as_quat()[[3, 0, 1, 2]]
        # Only expert-data augmentation supplies this temporary motion target.
        # Keep the original grasp diagnostic/observation untouched, and restore
        # its normal target immediately on the next non-perturbation query.
        perturbation = request.get('perturbation_pose')
        if perturbation is not None:
            perturbation = np.asarray(perturbation, dtype=float)
            if perturbation.shape != (7,) or not np.isfinite(perturbation).all():
                raise ValueError('Invalid perturbation pose')
            if not np.isclose(np.linalg.norm(perturbation[3:]), 1., atol=1.e-5):
                raise ValueError('Perturbation quaternion must be normalized')
            command_position, command_quaternion = perturbation[:3], perturbation[3:]
        self.set_pose(self.target_connection, np.r_[command_position, command_quaternion])
        self.world.notify_state_change(publish_changes=False)
        return dict(stage=stage, pregrasp_distance=pre_distance,
            command_goal_position=command_position.tolist(), command_goal_quaternion=command_quaternion.tolist(),
            perturbation_command=perturbation is not None, distance=float(np.linalg.norm(offset)),
            angle=float(angle), hand_pose=hand.tolist(), goal_position=goal_position.tolist(),
            goal_quaternion=goal_rotation.as_quat()[[3, 0, 1, 2]].tolist())

    def command(self, request):
        """One QP command from current measurements; no upright-only templates."""
        started = time.perf_counter()
        if not np.isclose(request['step_dt'], self.control_dt):
            raise ValueError('Giskard and RL control periods must match')
        diagnostic = self.sync(request)
        compile_seconds = 0.
        if not self.compiled:
            compile_start = time.perf_counter()
            self.executor.compile(self.chart)
            self.compiled = True
            self.initial_life_cycle = self.chart.life_cycle_state.data.copy()
            self.initial_observation = self.chart.observation_state.data.copy()
            compile_seconds = time.perf_counter()-compile_start
        self.chart.life_cycle_state.data[:] = self.initial_life_cycle
        self.chart.observation_state.data[:] = self.initial_observation
        self.chart.history.history.clear()
        self.executor.control_cycles = 0
        for node in self.chart.nodes:
            if isinstance(node, CartesianTask):
                node._forward_kinematics_binding.bind(self.world)
        before = np.array([joint.position for joint in self.arm])
        tick_start = time.perf_counter()
        self.executor.tick()
        tick_seconds = time.perf_counter()-tick_start
        delta = np.array([joint.position for joint in self.arm])-before
        previous = np.asarray(request['snapshot']['last_action'])
        old_target = np.clip(PARK+1.5*previous[3:], ARM_TARGET_LOWER, ARM_TARGET_UPPER)
        target = np.clip(old_target+delta, ARM_TARGET_LOWER, ARM_TARGET_UPPER)
        velocity = [self.world.state[d.id].velocity for d in (self.drive.x_velocity, self.drive.y_velocity, self.drive.yaw)]
        action = np.r_[np.clip(np.array(velocity)/[.15, .15, .4], -1, 1), (target-PARK)/1.5]
        if not np.isfinite(action).all():
            raise ValueError('Nonfinite Giskard command')
        return dict(id=request['id'], status='command_found', action=action.tolist(),
            planning_seconds=time.perf_counter()-started, compile_seconds=compile_seconds,
            tick_seconds=tick_seconds, diagnostic=diagnostic,
            initialization_seconds=self.initialization_seconds,
            source='giskard_qp_approach', command_version='giskard_velocity_integrated_v5')
