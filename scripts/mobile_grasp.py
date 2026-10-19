"""Run state-based GraspGen grasping with GARMI's existing CRAM/Giskard actions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np
import trimesh
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from sensor_msgs.msg import JointState

sys.path.insert(0, '/home/jovyan/cram-vrb-lab')
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, MovementType
from coraplex.locations.pose_validator import AreReachableBy
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.mixins import HasApproachesGraspPoses
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from semantic_digital_twin.datastructures.definitions import GripperState
from coraplex.execution_environment import real_robot, simulated_robot
from coraplex.view_manager import ViewManager
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.api import BodySpecification, Connection6DoFSpecification
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from cram_vrb_lab.robots.garmi.motions import GARMI_MOTION_MAPPINGS
from cram_vrb_lab.sim.scene_sync import SceneSyncClient, body_T_shape, SCENE_SYNC_POSE_TOPIC
from cram_vrb_lab.sim.scene_reset import SceneResetClient, cancel_motion
from grasp_adapter import GraspAdapter, load_candidates
from grasp_clearance import GraspClearance
from cram_vrb_lab.robots.garmi.joints import load_patched_urdf, JOINT_STATES_TOPIC
from cram_vrb_lab.robots.garmi.motions import GRIPPER_OPEN_TOLERANCE


def matrix(entity):
    return np.asarray(entity.to_np(), dtype=float)


def pose(transform, reference):
    return HomogeneousTransformationMatrix(data=transform, reference_frame=reference).to_pose()


@dataclass
class MobileGrasp:
    options: argparse.Namespace
    output: Path = field(default=Path('/workspace/results/mobile-grasp'))
    cache: Path = field(default=Path('/workspace/results/mobile-grasp'))
    events: list = field(default_factory=list)
    event_publisher: object = field(default=None, init=False)

    def record(self, event, **values):
        item = dict(event=event, time=time.time(), **values)
        encoded = json.dumps(item)
        self.events.append(item)
        print(encoded, flush=True)
        (self.output / f'{self.options.stage}-report.json').write_text(
            json.dumps(self.events, indent=2) + '\n')
        if self.event_publisher is not None:
            self.event_publisher.publish(String(data=encoded))

    def connect(self):
        self.output.mkdir(parents=True, exist_ok=True)
        rclpy.init()
        self.node = rclpy.create_node('fame_mobile_grasp')
        self.event_publisher = self.node.create_publisher(String, '/fame/task_event', 10)
        self.measured_joints = (0., {})
        self.joint_subscription = self.node.create_subscription(
            JointState, JOINT_STATES_TOPIC, self.receive_robot_joints, 10)
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self.world = fetch_world_from_service(node=self.node, timeout_seconds=120)
        self.synchronizer = WorldSynchronizer(_world=self.world, node=self.node)
        self.robot = self.world.get_semantic_annotations_by_type(Garmi)[0]
        self.robot.mobile_base.full_body_controlled = False
        self.context = Context(world=self.world, robot=self.robot, ros_node=self.node,
                               evaluate_conditions=False, alternative_motion_mappings=GARMI_MOTION_MAPPINGS)
        self.sync = SceneSyncClient(self.node)
        self.pose_received_at = 0.0
        self.pose_subscription = self.node.create_subscription(
            String, SCENE_SYNC_POSE_TOPIC, self.note_pose_update, 10)
        self.hand = self.world.get_body_by_name('right_fr3_hand')
        self.end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, self.robot)
        self.tool = self.end_effector.tool_frame
        self.mesh_path = Path('/workspace/cram-vrb-lab/cognitive_robot_abstract_machine/coraplex/resources/objects/bowl.stl')
        self.mesh = trimesh.load(self.mesh_path, force='mesh')
        self.predictions, self.scores = load_candidates(self.cache / 'bowl-grasps.npz', self.mesh_path)
        hand_to_tool = matrix(self.world.compute_forward_kinematics(root=self.hand, tip=self.tool))
        self.adapter = GraspAdapter(hand_to_tool,
            matrix(self.end_effector.front_facing_orientation.to_rotation_matrix())[:3, :3])
        self.record('connected', base=matrix(self.robot.root.global_pose).tolist(),
                    tcp=matrix(self.tool.global_pose).tolist(), hand_to_tcp=hand_to_tool.tolist())
        furniture = {}
        for body in self.world.bodies:
            if any(word in str(body.name) for word in ('chair', 'table', 'cabinet')) and len(body.collision):
                bounds = body.collision.as_bounding_box_collection_in_frame(self.world.root).bounding_box().to_array_bounds()
                furniture[str(body.name)] = [bounds.lower.tolist(), bounds.upper.tolist()]
        (self.output / 'furniture.json').write_text(json.dumps(furniture, indent=2) + '\n')

    def setup_object(self):
        offset_x, offset_y = self.options.object_offset
        position = [0.5 + offset_x, 7.2 + offset_y, 1.0]
        initial = HomogeneousTransformationMatrix.from_xyz_rpy(*position)
        name = 'graspgen_bowl.stl'
        if not self.world.is_kinematic_structure_entity_in_world_by_name(name):
            Bowl.get_annotation_specification(name,
                BodySpecification.mesh(name, str(self.mesh_path), parent_T_self=initial),
                parent_connection_specification=Connection6DoFSpecification()).spawn(self.world)
            time.sleep(1)
        self.body = self.world.get_body_by_name(name)
        self.annotation = next(a for a in self.world.get_semantic_annotations_by_type(Bowl) if a.root is self.body)
        self.world.move_branch(self.body, self.world.root)
        self.sync.detach(name)
        self.sync.place(name, position, [0, 0, 0, 1], mesh=str(self.mesh_path),
                        mass=0.058, collider='convexDecomposition', color=[0.2, 0.45, 0.8], track=True)
        self.record('spawn', acknowledgement=self.sync.apply(), requested_position=position,
                    offset_xy_m=list(self.options.object_offset))
        time.sleep(2)
        self.sync.pull(self.world)
        self.record('settled', object=self.object_pose(), output_dir=str(self.output))

    def object_pose(self):
        if time.monotonic() - self.pose_received_at > 1.0:
            raise RuntimeError('Isaac object measurements are stale')
        result = self.sync.tracked_poses().get(str(self.body.name))
        if result is None:
            raise RuntimeError('Isaac has not reported the tracked object pose')
        return result

    def note_pose_update(self, message):
        if 'graspgen_bowl_stl' in json.loads(message.data):
            self.pose_received_at = time.monotonic()

    def run_motion(self, label, action, timeout=75):
        self.record('motion_start', label=label)
        timed_out = threading.Event()
        def stop():
            timed_out.set()
            print('MOTION_TIMEOUT ' + label, flush=True)
            cancel_motion(self.context)
        timer = threading.Timer(timeout, stop)
        timer.daemon = True
        timer.start()
        try:
            with real_robot(collision_avoidance=True):
                execute_single(action, context=self.context).perform()
        finally:
            timer.cancel()
        if timed_out.is_set():
            raise TimeoutError(label)
        self.record('motion_done', label=label, tcp=matrix(self.tool.global_pose).tolist(),
                    base=matrix(self.robot.root.global_pose).tolist(), object=self.object_pose())

    def navigate(self, coordinates):
        x, y, yaw = coordinates
        target = HomogeneousTransformationMatrix.from_xyz_rpy(x, y, 0, yaw=yaw).to_pose()
        target.reference_frame = self.world.root
        self.run_motion('navigate', NavigateAction(target_location=target, keep_joint_states=True))
        actual = matrix(self.robot.root.global_pose)
        error = np.linalg.norm(actual[:2, 3] - [x, y])
        if error > 0.06:
            raise RuntimeError(f'Base stopped {error:.3f} m from its goal')

    def select(self):
        self.sync.pull(self.world)
        body_to_mesh = body_T_shape(self.body)
        world_to_mesh = matrix(self.body.global_pose) @ body_to_mesh
        support_height = min(trimesh.transform_points(self.mesh.vertices, world_to_mesh)[:, 2])
        gripper = trimesh.load(self.cache / 'panda-gripper.ply', force='mesh')
        approach = HasApproachesGraspPoses(approach_clearance=0.08, retreat_distance=0.12)
        clearance = GraspClearance(self.mesh, load_patched_urdf(),
                                   finger_travel=.04-GRIPPER_OPEN_TOLERANCE)
        clearance_results = []
        geometric = []
        for index, prediction in enumerate(self.predictions):
            if self.options.rank is not None and index != self.options.rank - 1:
                continue
            world_to_gripper = world_to_mesh @ prediction
            vertices = trimesh.transform_points(gripper.vertices, world_to_gripper)
            if world_to_gripper[2, 2] > -0.4 or vertices[:, 2].min() < support_height + 0.003:
                continue
            grasp = pose(self.adapter.to_grasp(prediction, body_to_mesh), self.body)
            poses = approach.grasp_pose_sequence(grasp, self.end_effector, grasp)
            distance = float(np.linalg.norm(matrix(poses[0])[:3,3]-matrix(poses[1])[:3,3]))
            checked = clearance.check(prediction, approach_distance=distance)
            clearance_results.append(dict(rank=index+1, **checked))
            if not checked['passed']:
                continue
            geometric.append(index)
        self.record('gripper_clearance_filter', results=clearance_results)
        self.record('geometry_filter', accepted_ranks=[i + 1 for i in geometric],
                    support_height=float(support_height), input_count=len(self.predictions))
        # The probe owns a separate world; planning must not publish hypothetical states.
        copied = deepcopy(self.world)
        robot = copied.get_semantic_annotation_by_id(self.robot.id)
        end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)
        body = copied.get_body_by_name(str(self.body.name))
        context = self.context.for_world(copied, robot)
        approach = HasApproachesGraspPoses(approach_clearance=0.08, retreat_distance=0.12)
        for index in geometric[:self.options.max_candidates]:
            transform = self.adapter.to_grasp(self.predictions[index], body_to_mesh)
            candidate = pose(transform, body)
            sequence = approach.grasp_pose_sequence(candidate, end_effector, candidate)
            self.record('probe_start', rank=index + 1, score=float(self.scores[index]))
            started = time.monotonic()
            try:
                with simulated_robot(collision_avoidance=True):
                    reachable = AreReachableBy(context=context, pose_sequence=sequence,
                        tip_link=end_effector.tool_frame)()
            except Exception as error:
                self.record('probe_error', rank=index + 1, error=repr(error))
                continue
            self.record('probe_done', rank=index + 1, reachable=bool(reachable), seconds=time.monotonic() - started)
            if reachable:
                self.selected_index = index
                self.selected = pose(transform, self.body)
                report = dict(rank=index + 1, score=float(self.scores[index]),
                    object_T_grasp=transform.tolist(), mesh_T_gripper=self.predictions[index].tolist(),
                    world_T_object=matrix(self.body.global_pose).tolist(),
                    base_pose=matrix(self.robot.root.global_pose).tolist())
                (self.output / 'selected.json').write_text(json.dumps(report, indent=2) + '\n')
                return
        raise RuntimeError('No reachable GraspGen candidate at this station')

    def receive_robot_joints(self, message):
        self.measured_joints = (time.monotonic(), dict(zip(message.name, message.position)))

    def measured_finger_positions(self):
        received, positions = self.measured_joints
        if time.monotonic()-received > .25:
            raise RuntimeError('Physical joint-state feedback is stale')
        return [float(positions[f'right_fr3_finger_joint{i}']) for i in [1,2]]

    def open_gripper_verified(self):
        # The motion chart can finish on a predicted state before the physical
        # fingers settle. Re-plan from measured feedback if the opening is short.
        for attempt in range(1, 4):
            self.run_motion('open_gripper', MoveGripperMotion(motion=GripperState.OPEN, gripper=Arms.RIGHT))
            time.sleep(.25)
            deadline = time.monotonic()+1.
            stable_since = None
            opened = False
            while True:
                positions = self.measured_finger_positions()
                if all(abs(value-.04) <= GRIPPER_OPEN_TOLERANCE for value in positions):
                    stable_since = stable_since or time.monotonic()
                    opened = time.monotonic()-stable_since >= .2
                else:
                    stable_since = None
                if opened or time.monotonic() >= deadline:
                    break
                time.sleep(.02)
            self.record('gripper_open_verified', positions_m=positions, passed=opened,
                        tolerance_m=GRIPPER_OPEN_TOLERANCE, attempt=attempt,
                        feedback_topic=JOINT_STATES_TOPIC)
            if opened:
                break
        if not opened:
            raise RuntimeError('Measured gripper aperture does not match the screened grasp geometry')

    def pick(self):
        self.open_gripper_verified()
        before = np.array(self.object_pose()['position'])
        approach = HasApproachesGraspPoses(approach_clearance=0.08, retreat_distance=0.12)
        _, _, lift = approach.grasp_pose_sequence(self.selected, self.end_effector, self.selected)
        # Freeze lift in map: the measured object must not move its own target.
        lift = self.world.transform(lift.to_homogeneous_matrix(), self.world.root).to_pose()
        self.run_motion('grasp', GraspingAction(object_designator=self.annotation, arm=Arms.RIGHT,
            grasp_pose=self.selected, approach_clearance=0.08, retreat_distance=0.12,
            final_approach_linear_velocity=0.04))
        self.run_motion('lift', MoveToolCenterPointMotion(lift, Arms.RIGHT,
            allow_gripper_collision=True, movement_type=MovementType.TRANSLATION, max_linear_velocity=0.04))
        samples = []
        for _ in range(21):
            samples.append(self.object_pose()['position'])
            time.sleep(0.1)
        minimum_height = float(np.array(samples)[:, 2].min() - before[2])
        passed = minimum_height >= 0.08
        self.record('physical_grasp_result', passed=passed, minimum_lift_m=minimum_height,
                    hold_seconds=2, automatic_attachment=False, positions=samples)
        if not passed:
            raise RuntimeError('The bowl was not physically held 8 cm above its support for 2 seconds')

    def transport(self):
        self.sync.pull(self.world)
        self.world.move_branch(self.body, self.tool)
        time.sleep(0.5)
        if self.options.carry == 'assisted':
            self.sync.sync_attachments(self.world, '/garmi')
            self.record('carry_attachment', mode='assisted', acknowledgement=self.sync.apply())
        self.run_motion('carry_pose', ParkArmsAction(Arms.BOTH))
        self.check_carry()
        self.navigate([0.25, 5.75, math.pi / 2])
        self.check_carry()
        self.navigate(self.options.place_base)
        self.check_carry()
        target = HomogeneousTransformationMatrix.from_xyz_rpy(*self.options.target).to_pose()
        target.reference_frame = self.world.root
        self.check_place(target)
        if self.options.carry == 'assisted':
            # Follow the semantic release inside PlaceAction; only its detach changes
            # the physics body back from kinematic to dynamic.
            self.carry_stop = threading.Event()
            def carry_sync():
                while not self.carry_stop.wait(0.05):
                    if self.sync.sync_attachments(self.world, '/garmi'):
                        self.sync.apply()
            self.carry_thread = threading.Thread(target=carry_sync, daemon=True)
            self.carry_thread.start()
        try:
            self.run_motion('place', PlaceAction(object_designator=self.body,
                target_location=target, arm=Arms.RIGHT, approach_clearance=0.08,
                retreat_distance=0.12), timeout=90)
        finally:
            if self.options.carry == 'assisted':
                self.carry_stop.set()
                self.carry_thread.join(timeout=12)
        time.sleep(2)
        actual = np.array(self.object_pose()['position'])
        horizontal_error = float(np.linalg.norm(actual[:2] - self.options.target[:2]))
        released = self.body.parent_kinematic_structure_entity is self.world.root
        passed = bool(released and horizontal_error < 0.05 and abs(actual[2] - self.options.target[2]) < 0.10)
        self.record('transport_result', passed=passed, carry_mode=self.options.carry,
                    object_position=actual.tolist(), horizontal_error_m=horizontal_error,
                    target=self.options.target, released=released)
        self.sync.pull(self.world)
        if not passed:
            raise RuntimeError('The bowl did not settle at the destination')
        self.run_motion('finish_park', ParkArmsAction(Arms.BOTH))

    def check_carry(self):
        expected = matrix(self.body.global_pose) @ body_T_shape(self.body)
        actual = np.array(self.object_pose()['position'])
        error = float(np.linalg.norm(expected[:3, 3] - actual))
        self.record('carry_check', error_m=error, mode=self.options.carry)
        if error > 0.08:
            raise RuntimeError('The object slipped from the expected grasp during transport')

    def check_place(self, target):
        copied = deepcopy(self.world)
        robot = copied.get_semantic_annotation_by_id(self.robot.id)
        end_effector = ViewManager.get_end_effector_view(Arms.RIGHT, robot)
        body = copied.get_body_by_name(str(self.body.name))
        held_grasp = end_effector.grasp_on(body)
        world_to_grasp = matrix(target) @ matrix(held_grasp)
        approach = HasApproachesGraspPoses(approach_clearance=0.08, retreat_distance=0.12)
        sequence = approach.grasp_pose_sequence(pose(world_to_grasp, copied.root),
            end_effector, held_grasp, reverse=True)
        self.record('place_probe_start')
        with simulated_robot(collision_avoidance=True):
            reachable = AreReachableBy(context=self.context.for_world(copied, robot),
                pose_sequence=sequence, tip_link=end_effector.tool_frame)()
        self.record('place_probe_done', reachable=bool(reachable))
        if not reachable:
            raise RuntimeError('The placement approach is not reachable from this station')

    def run(self):
        self.connect()
        if self.options.reset:
            cancel_motion(self.context)
            self.record('reset', result=SceneResetClient(self.node)())
            time.sleep(1)
        self.setup_object()
        if self.options.stage == 'inspect':
            return
        self.run_motion('park', ParkArmsAction(Arms.BOTH))
        self.navigate(self.options.base)
        self.select()
        if self.options.stage == 'plan':
            return
        self.pick()
        if self.options.stage == 'transport':
            self.transport()

    def close(self):
        self.executor.shutdown()
        self.thread.join(timeout=3)
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['inspect', 'plan', 'pick', 'transport'], default='transport')
    parser.add_argument('--base', nargs=3, type=float, default=[0.25, 6.45, math.pi / 2], metavar=('X', 'Y', 'YAW'))
    parser.add_argument('--rank', type=int)
    parser.add_argument('--max-candidates', type=int, default=12)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--carry', choices=['physical', 'assisted'], default='physical')
    parser.add_argument('--place-base', nargs=3, type=float, default=[0.90, 5.85, -0.85])
    parser.add_argument('--target', nargs=3, type=float, default=[1.6, 5.32, 0.82])
    parser.add_argument('--object-offset', nargs=2, type=float, default=[0.0, 0.0],
                        metavar=('DX', 'DY'), help='Bowl offset in world XY, in meters')
    parser.add_argument('--output-dir', type=Path, default=Path('/workspace/results/mobile-grasp'))
    options = parser.parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    demo = MobileGrasp(options, output=options.output_dir)
    try:
        demo.run()
        demo.record('completed', stage=options.stage)
    except Exception as error:
        demo.record('failed', error=repr(error))
        traceback.print_exc()
        raise
    finally:
        if rclpy.ok():
            demo.close()


if __name__ == '__main__':
    main()
