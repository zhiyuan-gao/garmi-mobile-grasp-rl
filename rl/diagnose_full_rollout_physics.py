"""Capture real reset states, plan offline, and replay admissible trajectories."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--cases', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--plans', help='Replay frozen plans instead of planning again')
parser.add_argument('--base-reference', choices=['legacy', 'link'], default='link')
parser.add_argument('--base-feedback', action='store_true')
parser.add_argument('--retime', action='store_true')
parser.add_argument('--case-index', type=int)
parser.add_argument('--contacts', action='store_true', help='Record per-step contact and grasp diagnostics')
parser.add_argument('--camera-eye', nargs=3, type=float)
parser.add_argument('--camera-lookat', nargs=3, type=float)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import copy
import json
import math
import os
import shutil
import subprocess
import time
import numpy as np
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from garmi_rl.config import configured_env, BODIES
from garmi_rl.giskard_io import snapshot_queries
from garmi_rl.joint_delta import encode_joint_target, is_joint_delta
from garmi_rl import mdp
from garmi_rl.trajectory_tracking import base_tracking_velocity, retime_plan
from scipy.spatial.transform import Rotation

torch.set_num_threads(4)
out = Path(args.out)
out.mkdir(exist_ok=False, parents=True)
cases = json.loads(Path(args.cases).read_text())
source_indices = list(range(len(cases))) if args.case_index is None else [args.case_index]
cases = [cases[i] for i in source_indices]
if args.enable_cameras and len(cases) != 1:
    raise ValueError('Recording requires a single case')
profile = yaml.safe_load(Path(args.config).read_text())
assert is_joint_delta(profile)
cfg, task = configured_env(profile, len(cases), 20261040)
cfg.actions.base.velocity_at_link = args.base_reference == 'link'
if args.enable_cameras:
    cfg.sim.render_interval = cfg.decimation*2
    x,y,z = cases[0]['object_state'][:3]
    cfg.viewer.eye = tuple(args.camera_eye or [x-.9,y-1.45,z+1.05])
    cfg.viewer.lookat = tuple(args.camera_lookat or [x-.12,y,z+.08])
env = ManagerBasedRLEnv(cfg, render_mode='rgb_array' if args.enable_cameras else None)
env.forbidden_sensors = cfg.forbidden_sensors
started = time.perf_counter()
video = None
try:
    env.reset()
    env.grasp_bank.restore(copy.deepcopy(cases))
    env._reset_idx = lambda ids: None
    if args.enable_cameras:
        import imageio.v2 as imageio
        from pxr import UsdGeom, UsdPhysics
        for prim in env.sim.stage.Traverse():
            if '/Table/collider_' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
                UsdGeom.Imageable(prim).CreatePurposeAttr('default')
        for _ in range(12):
            env.render()
        frame = env.render().copy()
        imageio.imwrite(out/'initial-frame.png',frame)
        video = imageio.get_writer(out/'replay.mp4',fps=10,macro_block_size=16)
        video.append_data(frame)
    obs = env.observation_manager.compute()['policy'].cpu().numpy()
    queries = snapshot_queries(env, obs, cases, list(range(len(cases))),
                               np.zeros(len(cases), bool), 'offline', 0)
    (out/'initial-queries.json').write_text(json.dumps(queries))
    native_environment = dict(os.environ, PYTHONPATH='/workspace/rl')
    native_environment.pop('PYTHONHOME', None)
    if args.plans:
        previous = json.loads(Path(args.plans).with_name('initial-queries.json').read_text())
        previous = [previous[i] for i in source_indices]
        assert len(previous) == len(queries)
        for old, new in zip(previous, queries):
            assert old['reference'] == new['reference']
            for key in ('robot_state', 'object_state', 'arm_position'):
                assert np.allclose(old['snapshot'][key], new['snapshot'][key], atol=1.e-5)
        shutil.copy2(args.plans, out/'plans.json')
    else:
        with (out/'planner.log').open('w') as log:
            subprocess.run([
                '/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python',
                '/workspace/rl/giskard_full_rollout.py',
                '--requests', str(out/'initial-queries.json'), '--out', str(out/'plans.json'),
                '--workers', str(args.workers)], stdout=log, stderr=subprocess.STDOUT,
                env=native_environment, check=True)
    plans = json.loads((out/'plans.json').read_text())['results']
    if args.plans:
        plans = [plans[i] for i in source_indices]
        (out/'plans.json').write_text(json.dumps(dict(results=plans)))
    assert len(plans) == len(cases)
    if args.retime:
        plans = [retime_plan(p,env.step_dt,cfg.actions.base.scale,cfg.actions.base.acceleration)
                 if p['admissible'] else p for p in plans]
        (out/'execution-plans.json').write_text(json.dumps(dict(results=plans)))
    active = np.array([p['admissible'] for p in plans])
    results = [dict(case=cases[i], planning={k: v for k, v in plan.items() if k != 'trajectory'},
                    physics={'reason': 'not_executed_invalid_plan'})
               for i, plan in enumerate(plans)]
    trace = []
    arm = env.action_manager.get_term('arm')
    for step in range(math.ceil(cfg.episode_length_s/env.step_dt)+2):
        if not active.any():
            break
        measured = env.scene['robot'].data.joint_pos[:, arm._joint_ids]
        targets = measured.clone()
        actions = torch.zeros((len(cases), 10), device=env.device)
        phase = env.fixed_finish.phase.cpu().numpy()
        measured_root = env.scene['robot'].data.root_link_state_w.cpu().numpy().copy()
        measured_root[:, :3] -= env.scene.env_origins.cpu().numpy()
        measured_yaw = Rotation.from_quat(measured_root[:, [4,5,6,3]]).as_euler('xyz')[:,2]
        for i in np.flatnonzero(active & (phase == 0)):
            trajectory = plans[i]['trajectory']
            node = trajectory[min(step, len(trajectory)-1)]
            targets[i] = targets.new_tensor(node['arm'])
            if step < len(trajectory):
                actions[i, :3] = actions.new_tensor(node['base_velocity'])/actions.new_tensor([.15, .15, .4])
            if args.base_feedback:
                previous = plans[i]['initial'] if step == 0 else trajectory[min(step-1,len(trajectory)-1)]
                # One full-speed acceleration period of the existing servo, independent of the grasp.
                response_time = max(v/a for v,a in zip(cfg.actions.base.scale, cfg.actions.base.acceleration))
                velocity = base_tracking_velocity(previous['base_pose'], node['base_pose'],
                    [*measured_root[i,:2], measured_yaw[i]], env.step_dt, response_time)
                actions[i,:3] = actions.new_tensor(velocity)/actions.new_tensor(cfg.actions.base.scale)
                actions[i,:3].clamp_(-1.,1.)
        actions[:, 3:] = encode_joint_target(measured, targets)
        with torch.no_grad():
            _, _, done, truncated, _ = env.step(actions)
        hp, _, _, _ = mdp.geometry(env)
        hand_positions = (hp-env.scene.env_origins).cpu().numpy()
        joints = env.scene['robot'].data.joint_pos[:, arm._joint_ids].cpu().numpy()
        roots = (env.scene['robot'].data.root_pos_w-env.scene.env_origins).cpu().numpy()
        root_state = env.scene['robot'].data.root_link_state_w.cpu().numpy().copy()
        root_state[:, :3] -= env.scene.env_origins.cpu().numpy()
        com_offset = env.scene['robot'].data.body_com_pos_b[:, 0].cpu().numpy()
        angles = Rotation.from_quat(root_state[:, [4,5,6,3]]).as_euler('xyz')
        d = env.diagnostics
        for i in np.flatnonzero(active):
            node = plans[i]['trajectory'][min(step, plans[i]['ticks']-1)]
            trace.append(dict(case=int(i), step=step, phase=int(phase[i]),
                root_state=root_state[i].tolist(), com_offset=com_offset[i].tolist(),
                root_rpy=angles[i].tolist(),
                commanded_velocity=env.action_manager.get_term('base').command[i].cpu().tolist(),
                servo_velocity=env.action_manager.get_term('base').velocity[i].cpu().tolist(),
                actual_arm=joints[i].tolist(), object_state=env.scene['object'].data.root_state_w[i].cpu().tolist(),
                hand_position=hand_positions[i].tolist(),
                yaw_error=float(np.arctan2(np.sin(angles[i,2]-node['base_pose'][2]),np.cos(angles[i,2]-node['base_pose'][2]))),
                arm_error=float(np.max(np.abs(joints[i]-node['arm']))),
                base_error=float(np.linalg.norm(roots[i, :2]-np.asarray(node['base_pose'])[:2])),
                hand_error=float(np.linalg.norm(hand_positions[i]-np.asarray(node['hand'])[:3, 3]))))
            if args.contacts:
                obj = env.scene['object'].data.root_state_w[i].clone()
                obj[:3] -= env.scene.env_origins[i]
                _,hq,gp,gq = mdp.geometry(env)
                pairs=[]
                for body,key in zip(BODIES,env.forbidden_sensors):
                    sensor=env.scene[key]
                    forces=sensor.data.force_matrix_w_history[i].norm(dim=-1).amax(dim=(0,1))
                    pairs.extend(dict(body=body,target=target,force=float(force))
                        for target,force in zip(sensor.cfg.filter_prim_paths_expr,forces) if force>.1)
                trace[-1].update(source_case_index=source_indices[i],phase_after=int(d['phase'][i]),
                    object_task_state=obj.cpu().tolist(),finger_force=d['finger_force'][i].cpu().tolist(),
                    contact=bool(d['contact'][i]),height=float(d['height'][i]),supported=bool(d['supported'][i]),
                    slip=float(d['slip'][i]),gripper_width=float(d['gripper_width'][i]),
                    distance=float(d['distance'][i]),angle=float(d['angle'][i]),
                    goal_position=(gp[i]-env.scene.env_origins[i]).cpu().tolist(),
                    goal_quaternion=gq[i].cpu().tolist(),hand_quaternion=hq[i].cpu().tolist(),
                    arm_target=arm.processed_actions[i].cpu().tolist(),contact_targets=pairs,
                    plan_stage=node.get('stage'),trajectory_fraction=min((step+1)/plans[i]['ticks'],1.))
        ended = (done | truncated).cpu().numpy() & active
        for i in np.flatnonzero(ended):
            reason = next((k for k in ('collision', 'dropped', 'out_of_bounds', 'success', 'attempt_failed')
                           if bool(d[k][i])), 'timeout')
            contact_targets = []
            for body, key in zip(BODIES, env.forbidden_sensors):
                sensor = env.scene[key]
                forces = sensor.data.force_matrix_w_history[i].norm(dim=-1).amax(dim=(0, 1))
                contact_targets.extend(dict(body=body, target=target, force=float(force))
                    for target, force in zip(sensor.cfg.filter_prim_paths_expr, forces) if force > .1)
            results[i]['physics'] = dict(reason=reason, seconds=(step+1)*env.step_dt,
                height=float(d['height'][i]), contact=bool(d['contact'][i]),
                supported=bool(d['supported'][i]), slip=float(d['slip'][i]),
                hold_seconds=float(env.grasp_state['hold'][i]),
                handover_distance=float(d['handover_distance'][i]),
                handover_trigger=int(d['handover_trigger'][i]), contact_targets=contact_targets)
        active &= ~ended
        if video is not None and ((step+1)%2==0 or not active.any()):
            frame=env.render().copy()
            video.append_data(frame)
        if step % 100 == 0:
            print('REPLAY', step, int(active.sum()), flush=True)
    assert not active.any(), 'Unfinished physical replay'
    if video is not None:
        video.close()
        video=None
        imageio.imwrite(out/'final-frame.png',frame)
    (out/'tracking.json').write_text(json.dumps(trace))
    (out/'summary.json').write_text(json.dumps(dict(status='complete', results=results,
        wall_seconds=time.perf_counter()-started, config=args.config,
        cases=args.cases, control_dt=env.step_dt, collision_sampling_dt=.005,
        base_reference=args.base_reference,
        base_feedback=args.base_feedback,
        retimed=args.retime,
        source_case_indices=source_indices,
        camera=dict(eye=list(cfg.viewer.eye),lookat=list(cfg.viewer.lookat)) if args.enable_cameras else None,
        replay=('Base trajectory feedforward plus position feedback' if args.base_feedback else 'Open-loop base velocities')
               + '; measured-to-planned joint increments; unchanged fixed finish'),
        indent=2))
    print('FULL_ROLLOUT_DONE', [r['physics']['reason'] for r in results], flush=True)
finally:
    if video is not None:
        video.close()
    env.close()
    app.close()
