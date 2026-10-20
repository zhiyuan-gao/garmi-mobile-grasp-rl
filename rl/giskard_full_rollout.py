"""Offline whole-approach rollout using the installed Giskard controller.

The measured state is imported once. Subsequent ticks advance the planning
world, without resetting controller history or importing simulator feedback.
This is a diagnostic, not a new expert-data collection backend.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from garmi_rl.giskard_expert import ApproachExpert
from garmi_rl.expert_actions import ARM_TARGET_LOWER, ARM_TARGET_UPPER
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianTask


def reset_goal(expert):
    expert.chart.life_cycle_state.data[:] = expert.initial_life_cycle
    expert.chart.observation_state.data[:] = expert.initial_observation
    expert.chart.history.history.clear()
    for node in expert.chart.nodes:
        if isinstance(node, CartesianTask):
            node._forward_kinematics_binding.bind(expert.world)


def state_record(expert):
    return dict(
        base_pose=[float(expert.world.state[d.id].position)
                   for d in (expert.drive.x, expert.drive.y, expert.drive.yaw)],
        base_velocity=[float(expert.world.state[d.id].velocity)
                       for d in (expert.drive.x_velocity, expert.drive.y_velocity, expert.drive.yaw)],
        arm=[float(j.position) for j in expert.arm],
        hand=expert.hand.global_transform.to_np().tolist())


def rollout(expert, request, subdivisions=10):
    started = time.perf_counter()
    assert np.isclose(request['step_dt'], expert.control_dt)
    diagnostic = expert.sync(request)
    if not expert.compiled:
        expert.executor.compile(expert.chart)
        expert.compiled = True
        expert.initial_life_cycle = expert.chart.life_cycle_state.data.copy()
        expert.initial_observation = expert.chart.observation_state.data.copy()
    reset_goal(expert)
    expert.executor.control_cycles = 0
    goal = np.asarray(diagnostic['goal_position'])
    quaternion = np.asarray(diagnostic['goal_quaternion'])
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
    pre = goal-rotation.apply([0., 0., .08])
    stage = diagnostic['stage']
    initial = state_record(expert)
    local = np.asarray(initial['hand'])[:3, :3].T @ (goal-np.asarray(initial['hand'])[:3, 3])
    frame_error = float(np.max(np.abs(local-np.asarray(request['snapshot']['observation'][30:33]))))
    assert frame_error < 3.e-4, ('Initial planning/physics mismatch', frame_error)
    path, first_collision, minimum = [], None, None
    first_limit_violation = None
    manager = expert.world.collision_manager

    def check(tick, fraction):
        nonlocal first_collision, minimum
        for contact in manager.compute_collisions().contacts:
            distance = float(contact.distance)
            if minimum is None or distance < minimum['distance']:
                minimum = dict(tick=tick, fraction=fraction, distance=distance,
                               a=str(contact.body_a.name), b=str(contact.body_b.name))
            if distance < -1.e-6 and first_collision is None:
                first_collision = dict(tick=tick, fraction=fraction, distance=distance,
                                       a=str(contact.body_a.name), b=str(contact.body_b.name))

    check(0, 0.)
    aligned = 0
    failure = None
    distance, angle = diagnostic['distance'], diagnostic['angle']
    maximum_ticks = round(request['approach_seconds']/expert.control_dt)
    for tick in range(maximum_ticks):
        before = expert.world.state.positions.copy()
        try:
            expert.executor.tick()
        except Exception as error:
            failure = dict(error=repr(error), traceback=traceback.format_exc(), tick=tick+1)
            break
        after = expert.world.state.positions.copy()
        # Check the interpolated joint/base path at the simulator's 5 ms period.
        # These are discrete geometric checks, not a continuous collision proof.
        for fraction in np.linspace(1/subdivisions, 1., subdivisions):
            expert.world.state.positions[:] = before+(after-before)*fraction
            expert.world.notify_state_change(publish_changes=False)
            check(tick+1, float(fraction))
        expert.world.state.positions[:] = after
        expert.world.notify_state_change(publish_changes=False)
        record = state_record(expert)
        joints = np.asarray(record['arm'])
        violations = np.maximum(ARM_TARGET_LOWER-joints,joints-ARM_TARGET_UPPER)
        if np.max(violations) > 1.e-6 and first_limit_violation is None:
            first_limit_violation = dict(tick=tick+1,joint=int(np.argmax(violations))+1,
                violation_rad=float(np.max(violations)))
        hand = np.asarray(record['hand'])
        distance = float(np.linalg.norm(hand[:3, 3]-goal))
        angle = float((rotation.inv()*Rotation.from_matrix(hand[:3, :3])).magnitude())
        record.update(stage=stage, distance=distance, angle=angle)
        path.append(record)
        if stage == 'pregrasp' and np.linalg.norm(hand[:3, 3]-pre) < .015 and angle < .15:
            stage = 'approach_grasp'
            expert.set_pose(expert.target_connection, np.r_[goal, quaternion])
            expert.world.notify_state_change(publish_changes=False)
            reset_goal(expert)
        aligned = aligned+1 if distance < .015 and angle < .15 else 0
        if aligned >= 2:
            break
    reached = aligned >= 2
    return dict(id=request['id'], case_id=request.get('initial_case', {}).get('id'),
                reference=request['reference'], status='planning_failed' if failure else 'complete',
                failure=failure, reached=reached,
                admissible=reached and first_collision is None and failure is None and first_limit_violation is None,
                first_limit_violation=first_limit_violation,
                first_collision=first_collision, minimum_clearance=minimum,
                ticks=len(path), simulated_seconds=len(path)*expert.control_dt,
                planning_seconds=time.perf_counter()-started, initial=initial,
                frame_error=frame_error, final_distance=distance, final_angle=angle,
                final_stage=stage, trajectory=path)


class Worker:
    expert = None

    @classmethod
    def initialize(cls):
        cls.expert = ApproachExpert()

    @classmethod
    def run(cls, request):
        try:
            result = rollout(cls.expert, request)
        except Exception as error:
            result = dict(id=request['id'], reference=request['reference'],
                          status='error', admissible=False, error=repr(error),
                          traceback=traceback.format_exc())
        print('ROLLOUT', request['id'], {k: v for k, v in result.items()
              if k in ('status', 'reached', 'admissible', 'ticks', 'error')}, flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    queries = json.loads(Path(args.requests).read_text())
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('fork'), initializer=Worker.initialize) as pool:
        for result in pool.map(Worker.run, queries):
            results.append(result)
            Path(args.out).write_text(json.dumps(dict(results=results,
                seconds=time.perf_counter()-started, status='planning')))
    Path(args.out).write_text(json.dumps(dict(results=results,
        seconds=time.perf_counter()-started, status='complete')))


if __name__ == '__main__':
    main()
