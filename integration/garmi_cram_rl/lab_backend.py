"""One persistent training scene controlled alternately by CRAM and the policy.

Import after AppLauncher. Physics, observations, action decoding, contacts and
the fixed finishing controller are the original training implementations.
"""
import json
from collections import deque
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from garmi_rl.reference_position_filter import ReferencePositionFilter
from garmi_cram_rl.contract import MapFrame, validate_start, workspace_path
from garmi_cram_rl.policy import Policy


class LabBackend:
    def __init__(self, bundle, base_map, bowl_map, render=False, full_scene=False, configure=None):
        from isaaclab.envs import ManagerBasedRLEnv
        from garmi_rl.config import configured_env
        from garmi_rl import mdp
        self.mdp = mdp
        self.bundle, self.frame, self.render = bundle, MapFrame(), render
        self.profile = yaml.safe_load((bundle.path / 'profile.yaml').read_text())
        self.task = self.profile['garmi_task']
        cfg, _ = configured_env(self.profile, 1, 12345)
        self.full_scene = full_scene
        if full_scene:
            cfg.scene.table.spawn.usd_path = str(bundle.workspace / 'integration/assets/apartment.usda')
        if configure is not None:
            configure(cfg)
        cfg.viewer.eye = (-1.4, -.9, 2.5)
        cfg.viewer.lookat = (.65, -.1, 1.1)
        cfg.viewer.resolution = (960, 640)
        self.env = ManagerBasedRLEnv(cfg, render_mode='rgb_array' if render else None)
        self.env.forbidden_sensors = cfg.forbidden_sensors
        self.env.reset()
        if render and not full_scene:
            from pxr import UsdGeom, UsdPhysics
            for prim in self.env.sim.stage.Traverse():
                if '/Table/collider_' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
                    UsdGeom.Imageable(prim).CreatePurposeAttr('default')
        self.robot, self.object = self.env.scene['robot'], self.env.scene['object']
        self.arm = self.robot.find_joints('right_fr3_joint[1-7]')[0]
        self.fingers = self.robot.find_joints('right_fr3_finger_joint.*')[0]
        self.base_term = self.env.action_manager.get_term('base')
        self.policy = Policy(bundle)
        self.mask = ReferencePositionFilter(self.task['kitchen']['mask'],
            self.task['multi_object']['manifest'], bundle.workspace, device='cpu')
        bank = json.loads((bundle.path / 'reference-bank.json').read_text())
        self.references = {r['id']: r for r in bank['objects'][0]['references']}
        metadata = json.loads((bundle.workspace / 'rl/assets/manifest.json').read_text())
        self.fixed_positions = metadata['frozen_joint_positions']
        self.targets = self.robot.data.joint_pos.clone()
        self.command_velocity = torch.zeros_like(self.targets)
        self.base_command = np.zeros(3)
        self.command_time = self.base_time = -float('inf')
        self.reference = None
        self.control_step = 0
        self.last_reason = None
        self.last_measurements = None
        self.arm_samples = deque(maxlen=11)
        # Initial scene setup happens once. No reset, teleport or attachment at handover.
        self._initial_pose(base_map, bowl_map)

    def camera(self, eye, target):
        self.camera_map = (list(eye), list(target))
        if self.render:
            self.env.sim.set_camera_view(self.frame.position_to_task(eye),
                                         self.frame.position_to_task(target))

    def _initial_pose(self, base_map, bowl_map):
        root = self.robot.data.root_state_w.clone()
        root[0, :3] = root.new_tensor(self.frame.position_to_task([*base_map[:2], .0259]))
        root[0, 3:7] = root.new_tensor(self.frame.quaternion_to_task(
            Rotation.from_euler('z', base_map[2]).as_quat()[[3, 0, 1, 2]]))
        root[0, 7:] = 0
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(root[:, 7:])
        state = self.object.data.root_state_w.clone()
        state[0, :3] = state.new_tensor(self.frame.position_to_task(
            [*bowl_map, self.mask.data['bowl_origin_z_m']]))
        state[0, 3:7] = state.new_tensor(self.frame.quaternion_to_task([1., 0., 0., 0.]))
        state[0, 7:] = 0
        self.object.write_root_pose_to_sim(state[:, :7])
        self.object.write_root_velocity_to_sim(state[:, 7:])
        self.env.grasp_state['initial_z'][:] = state[:, 2]
        self.env.sim.forward()
        self.env.scene.update(0.)
        self.targets[:] = self.robot.data.joint_pos

    def snapshot(self):
        if self.reference is not None:
            diagnostics = self.mdp.diagnostics(self.env)
            self.last_measurements = {name: float(diagnostics[name][0]) for name in
                                      ('height', 'slip', 'contact', 'supported', 'collision')}
            self.last_measurements['hold_seconds'] = float(self.env.grasp_state['hold'][0])
        root = self.robot.data.root_state_w[0].detach().cpu().numpy()
        obj = self.object.data.root_state_w[0].detach().cpu().numpy()
        base_q = self.frame.quaternion_to_map(root[3:7])
        object_q = self.frame.quaternion_to_map(obj[3:7])
        joints = dict(self.fixed_positions)
        joints.update(zip(self.robot.joint_names, self.robot.data.joint_pos[0].cpu().tolist()))
        joints.update({name: 0. for name in ('o1_motor_1', 'o1_motor_2',
            'left_fr3_finger_joint1', 'left_fr3_finger_joint2',
            'front_left_wheel_joint', 'front_right_wheel_joint',
            'rear_left_wheel_joint', 'rear_right_wheel_joint') if name not in joints})
        velocities = dict(zip(self.robot.joint_names, self.robot.data.joint_vel[0].cpu().tolist()))
        arm_position = self.robot.data.joint_pos[0, self.arm].cpu().tolist()
        if not self.arm_samples or self.arm_samples[-1][0] != self.control_step:
            self.arm_samples.append((self.control_step, arm_position))
        joint_range = np.ptp(np.asarray([sample[1] for sample in self.arm_samples]), axis=0)
        local_velocity = torch.cat((self.robot.data.root_lin_vel_b[0, :2],
                                    self.robot.data.root_ang_vel_b[0, 2:]))
        return dict(base_map_position=self.frame.position_to_map(root[:3]).tolist(),
            base_map_quaternion_wxyz=base_q.tolist(),
            base_map_yaw=float(Rotation.from_quat(base_q[[1, 2, 3, 0]]).as_euler('xyz')[2]),
            object_map_position=self.frame.position_to_map(obj[:3]).tolist(),
            object_map_quaternion_wxyz=object_q.tolist(),
            object_map_euler=Rotation.from_quat(object_q[[1, 2, 3, 0]]).as_euler('xyz').tolist(),
            object_velocity=obj[7:].tolist(), full_scene=self.full_scene,
            base_velocity=local_velocity.cpu().tolist(),
            arm_position=arm_position,
            joint_settling_range=joint_range.tolist(),
            settling_window_seconds=(self.arm_samples[-1][0]-self.arm_samples[0][0]) * self.env.step_dt,
            joint_velocity=self.robot.data.joint_vel[0, self.arm].cpu().tolist(),
            joints=joints, velocities=velocities,
            reference=self.reference, phase=int(self.env.fixed_finish.phase[0]),
            grasp_measurements=self.last_measurements,
            simulation_seconds=self.control_step * self.env.step_dt)

    def admit(self, reference):
        self.reference = validate_start(self.snapshot(), self.task, self.mask, reference)
        if bool(self.mdp.diagnostics(self.env)['collision'][0]):
            raise ValueError('Robot is already in forbidden contact')

    def clear_classical_commands(self):
        self.base_command[:] = 0.
        self.command_velocity.zero_()
        self.command_time = self.base_time = -float('inf')

    def begin(self, reference):
        self.clear_classical_commands()
        env = self.env
        # Reset controller memory only, preserving all measured physical state.
        env.action_manager.reset()
        env.fixed_finish.reset(torch.tensor([0], device=env.device))
        env.episode_length_buf.zero_()
        self.base_term.velocity.zero_()
        ref = self.references[self.reference]
        env.grasp_pos_o[:] = env.grasp_pos_o.new_tensor(ref['position'])
        env.grasp_quat_o[:] = env.grasp_quat_o.new_tensor(ref['quaternion'])
        index = self.mask.data['reference_ids'].index(self.reference)
        env.grasp_bank.reference_ids[:] = index
        for name in ('hold', 'ever_lifted', 'anchor', 'held_last'):
            env.grasp_state[name].zero_()
        env.grasp_state['initial_z'][:] = self.mask.data['bowl_origin_z_m']
        env.diagnostic_step = -1
        self.last_reason = None
        self.last_measurements = None

    def _physics(self, policy):
        env = self.env
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            if policy:
                env.action_manager.apply_action()
            else:
                self.base_term.apply_actions()
                self.robot.set_joint_position_target(self.targets)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        env.common_step_counter += 1
        self.control_step += 1
        if self.render and self.control_step % 2 == 0:
            env.sim.render()

    def step_policy(self):
        env = self.env
        observation = self.mdp.observation(env)
        action = self.policy(observation.detach().cpu()).to(env.device)
        env.action_manager.process_action(action)
        self._physics(policy=True)
        env.episode_length_buf += 1
        diagnostics = self.mdp.diagnostics(env)
        self.last_measurements = {name: float(diagnostics[name][0]) for name in
                                  ('height', 'slip', 'contact', 'supported', 'collision')}
        self.last_measurements['hold_seconds'] = float(env.grasp_state['hold'][0])
        for reason in ('collision', 'dropped', 'out_of_bounds', 'success', 'attempt_failed'):
            if bool(diagnostics[reason][0]):
                self.last_reason = reason
                return reason
        if float(env.episode_length_buf[0]) * env.step_dt >= env.cfg.episode_length_s:
            self.last_reason = 'timeout'
            return 'timeout'
        return None

    def hold(self):
        self.clear_classical_commands()
        self.targets[:] = self.robot.data.joint_pos
        if self.last_reason == 'success':
            self.targets[:, self.arm] = self.env.action_manager.get_term('arm').processed_actions
        self.targets[:, self.fingers] = self.env.action_manager.get_term('gripper').processed_actions
        self.base_term._processed_actions.zero_()

    def return_to_cram(self):
        self.clear_classical_commands()
        # Preserve the closing target: reseeding fingers from measured opening drops the bowl.
        self.targets[:, self.arm] = self.robot.data.joint_pos[:, self.arm]

    def set_joint_velocity(self, names, values, now):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError('Invalid Giskard joint velocity message')
        self.command_velocity.zero_()
        for name, value in zip(names, values):
            if name in self.robot.joint_names:
                self.command_velocity[0, self.robot.joint_names.index(name)] = value
            elif abs(value) > .01:
                raise ValueError(f'Training asset holds this joint fixed: {name}')
        self.command_time = now

    def set_base_velocity(self, value, now):
        command = np.asarray(value, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError('Invalid base velocity')
        # Native cmd_vel is already scaled by Giskard. Policy scaling applies
        # only in the original RL action term during exclusive RL ownership.
        self.base_command = command.copy()
        self.base_time = now

    def set_gripper(self, value):
        if not np.isfinite(value) or not 0. <= value <= .04:
            raise ValueError('Finger travel must be within [0, .04] metres')
        self.targets[:, self.fingers] = value

    def step_classical(self, now, holding=False):
        base = np.zeros(3) if holding or now - self.base_time > .5 else self.base_command
        self.base_term._processed_actions[:] = self.base_term.velocity.new_tensor(base)
        if not holding and now - self.command_time <= .5:
            measured = self.robot.data.joint_pos
            next_target = self.targets + self.command_velocity * self.env.step_dt
            next_target = torch.maximum(torch.minimum(next_target, measured + .02), measured - .02)
            moving = self.command_velocity.abs() > 1.e-6
            self.targets[:] = torch.where(moving, next_target, self.targets)
            # The physical finger limits include 0 and .04; soft limits would
            # turn a closed target into .001 m and weaken the handover grip.
            limits = self.robot.data.joint_pos_limits
            self.targets[:] = self.targets.clamp(limits[:, :, 0], limits[:, :, 1])
        self._physics(policy=False)

    def close(self):
        self.env.close()
