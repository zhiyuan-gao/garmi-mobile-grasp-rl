"""Lightweight Isaac-side contract for the CPU Giskard expert."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys

import numpy as np

COMMAND_VERSION = 'giskard_velocity_integrated_v5'
APPROACH_CLEARANCE = .08


def advance_stages(stages, observations):
    """Advance only; call on every measured step, even without a label query.

    This is teacher context, not a new policy input or an executed learner action.
    The goal quaternion in the observation rotates goal coordinates into the hand.
    """
    x = np.asarray(observations)
    w, qx, qy, qz = x[:, 33:37].T
    axis = np.stack((2*(qx*qz+w*qy), 2*(qy*qz-w*qx), 1-2*(qx*qx+qy*qy)), axis=1)
    pre_distance = np.linalg.norm(x[:, 30:33]-APPROACH_CLEARANCE*axis, axis=1)
    angle = 2*np.arccos(np.clip(np.abs(w), 0., 1.))
    return np.asarray(stages, dtype=bool) | ((pre_distance < .015) & (angle < .15))


def entering_finish(env, observations):
    x = np.asarray(observations)
    angle = 2*np.arccos(np.clip(np.abs(x[:, 33]), 0., 1.))
    aligned = (np.linalg.norm(x[:, 30:33], axis=1) < env.cfg.fixed_finish['alignment_distance']) & (
        angle < env.cfg.fixed_finish['alignment_angle'])
    finish = env.fixed_finish
    return ((finish.phase_steps.cpu().numpy()+1 >= finish.ticks('approach_seconds')) |
        (aligned & (finish.aligned_steps.cpu().numpy()+1 >= finish.ticks('alignment_seconds'))))


def snapshot_queries(env, observations, cases, selected, stages, prefix, step):
    """Capture measured state, previous target, timing, and teacher stage together."""
    robot = env.scene['robot']
    arm = robot.find_joints('right_fr3_joint[1-7]')[0]
    roots = robot.data.root_state_w.clone()
    roots[:, :3] -= env.scene.env_origins
    objects = env.scene['object'].data.root_state_w.clone()
    objects[:, :3] -= env.scene.env_origins
    arrays = dict(robot_state=roots.cpu().numpy(), object_state=objects.cpu().numpy(),
        arm_position=robot.data.joint_pos[:, arm].cpu().numpy(),
        arm_velocity=robot.data.joint_vel[:, arm].cpu().numpy(),
        base_servo_velocity=env.action_manager.get_term('base').velocity.cpu().numpy(),
        last_action=env.action_manager.action.cpu().numpy(), observation=np.asarray(observations))
    if getattr(env.cfg, 'joint_delta_actions', False):
        from .expert_actions import PARK
        # Planner history needs the actual commanded joint target. The policy's
        # previous increment must not be misinterpreted as an absolute target.
        target = env.action_manager.get_term('arm').processed_actions.cpu().numpy()
        arrays['policy_last_action'] = arrays['last_action'].copy()
        arrays['last_action'] = np.concatenate((arrays['last_action'][:, :3], (target-PARK)/1.5), axis=1)
    phase_steps = env.fixed_finish.phase_steps.cpu().numpy()
    aligned = env.fixed_finish.aligned_steps.cpu().numpy()
    episode_steps = env.episode_length_buf.cpu().numpy()
    return [dict(id=f'{prefix}-c{i:04d}-s{step:04d}', reference=cases[i]['reference'],
        initial_case=cases[i], case=i, step=step,
        approach_stage='approach_grasp' if stages[i] else 'pregrasp',
        snapshot=dict(object=cases[i]['object'], reference=cases[i]['reference'],
            **{key: value[i].tolist() for key, value in arrays.items()},
            elapsed_seconds=float(phase_steps[i]*env.step_dt), aligned_steps=int(aligned[i]),
            episode_steps=int(episode_steps[i])),
        approach_seconds=env.cfg.fixed_finish['approach_seconds'], step_dt=env.step_dt)
        for i in selected]


class GiskardClient:
    """Persistent separate interpreter; no ROS or full navigation task invocation."""

    def __init__(self, logfile, workers=8):
        self.log = Path(logfile).open('w')
        self.process = subprocess.Popen([
            '/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python',
            '/workspace/rl/giskard_expert_service.py', '--serve', '--workers', str(workers)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, env=dict(os.environ, PYTHONPATH='/workspace/rl',
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        assert self._receive()['ready']
        self.query_seconds, self.batch_seconds, self.compile_seconds = [], [], []
        self.max_frame_error = 0.

    def _receive(self):
        if not select.select([self.process.stdout], [], [], 300)[0]:
            raise TimeoutError('Giskard service did not answer in 300 seconds')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Giskard service stopped: {self.process.poll()}')
        return json.loads(line)

    def query(self, requests):
        if not requests:
            return []
        self.process.stdin.write(json.dumps(requests)+'\n')
        self.process.stdin.flush()
        response = self._receive()
        self.batch_seconds.append(response['batch_seconds'])
        assert len(response['results']) == len(requests)
        for request, result in zip(requests, response['results']):
            assert result['id'] == request['id']
            self.query_seconds.append(result['planning_seconds'])
            if result['status'] != 'command_found':
                continue
            assert result['command_version'] == COMMAND_VERSION
            assert np.isfinite(result['action']).all() and len(result['action']) == 10
            if result['compile_seconds']:
                self.compile_seconds.append(result['compile_seconds'])
            diagnostic = result['diagnostic']
            hand = np.array(diagnostic['hand_pose'])
            delta = hand[:3, :3].T @ (np.array(diagnostic['goal_position'])-hand[:3, 3])
            error = float(np.max(np.abs(delta-np.array(request['snapshot']['observation'][30:33]))))
            self.max_frame_error = max(self.max_frame_error, error)
            assert error < 3.e-4, ('Planning/physics frame mismatch', error)
        return response['results']

    def timing(self):
        return dict(queries=len(self.query_seconds),
            mean_query_ms=1000*float(np.mean(self.query_seconds)) if self.query_seconds else None,
            p95_query_ms=1000*float(np.percentile(self.query_seconds,95)) if self.query_seconds else None,
            total_batch_seconds=sum(self.batch_seconds), compile_seconds=self.compile_seconds,
            max_frame_error=self.max_frame_error)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=60)
        self.log.close()
