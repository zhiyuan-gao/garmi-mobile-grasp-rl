"""Initialize at a cup grasp, then test the unchanged physical close/lift/hold.

Diagnostic state initialization is NOT a planned/executable approach. No object
attachment, friction change, force change, or success-threshold change is used.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--case-index', type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import collections
import hashlib
import json
import math
import time
import traceback
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import subtract_frame_transforms, quat_error_magnitude, quat_apply_inverse
from garmi_rl.config import configured_env, BODIES
from garmi_rl.diagnostic_ik import DiagnosticIK
from garmi_rl.expert_actions import ARM_TARGET_LOWER, ARM_TARGET_UPPER
from garmi_rl import mdp

torch.set_num_threads(4)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
profile = yaml.safe_load(Path(args.config).read_text())
bank = json.loads(Path(profile['garmi_task']['multi_object']['manifest']).read_text())['objects'][0]
refs = bank['references']
assert bank['name'] == 'cup' and len(refs) == 4
assert [r['name'] for r in profile['garmi_task']['kitchen']['object_regions']] == ['left']
specs = [dict(reference=r['id'], position_index=k, object_xy=[x,y]) for r in refs
         for k,(x,y) in enumerate(( (x,y) for x in (.90,.96) for y in (-.15,0.,.15,.30,.45) ))]
if args.case_index is not None:
    specs = [specs[args.case_index]]
if args.enable_cameras and len(specs) != 1:
    raise ValueError('Use --case-index for video')
cfg, _ = configured_env(profile, len(specs), 2026102021)
if args.enable_cameras:
    x,y = specs[0]['object_xy']
    cfg.sim.render_interval = cfg.decimation*2
    cfg.viewer.eye = (x-.9,y-1.45,1.996)
    cfg.viewer.lookat = (x-.12,y,1.026)
env = ManagerBasedRLEnv(cfg, render_mode='rgb_array' if args.enable_cameras else None)
env.forbidden_sensors = cfg.forbidden_sensors
video = None
started = time.perf_counter()
try:
    env.reset()
    n = len(specs)
    robot, obj = env.scene['robot'], env.scene['object']
    arm = robot.find_joints('right_fr3_joint[1-7]')[0]
    fingers = env.finger_joint_ids
    cases = env.grasp_bank.cases()
    reference_map = {r['id']:r for r in refs}
    diagnostic_roots = []
    seeds = []
    for i,s in enumerate(specs):
        ref = reference_map[s['reference']]
        seed = ref['validation_seed']
        case = cases[i]
        case.update(reference=s['reference'], id=f"hold-{s['reference']}-{s['position_index']:02}")
        case['object_state'] = list(seed['object_state'])
        case['object_state'][:2] = s['object_xy']
        # Restore with a legal episode base first, then explicitly initialize
        # the diagnostic base to preserve the old validated relative geometry.
        case['robot_state'] = list(seed['robot_state'])
        root = list(seed['robot_state'])
        root[0] += s['object_xy'][0]-seed['object_state'][0]
        root[1] += s['object_xy'][1]-seed['object_state'][1]
        diagnostic_roots.append(root)
        seeds.append(seed['joint_position'])
    env.grasp_bank.restore(cases)
    roots = torch.tensor(diagnostic_roots, device=env.device)
    roots[:,:3] += env.scene.env_origins
    robot.write_root_pose_to_sim(roots[:,:7])
    robot.write_root_velocity_to_sim(torch.zeros(n,6,device=env.device))
    env.sim.forward(); env.scene.update(0.)
    hp,hq,gp,gq = mdp.geometry(env)
    p,q = subtract_frame_transforms(robot.data.root_pos_w,robot.data.root_quat_w,gp,gq)
    limits = robot.data.soft_joint_pos_limits[:,arm].clone()
    limits[:,:,0] = torch.maximum(limits[:,:,0], limits.new_tensor(ARM_TARGET_LOWER))
    limits[:,:,1] = torch.minimum(limits[:,:,1], limits.new_tensor(ARM_TARGET_UPPER))
    seeds = torch.tensor(seeds,device=env.device)[:,arm]
    solved = DiagnosticIK(env.device).solve(p,q,seeds,limits)
    joints = robot.data.joint_pos.clone()
    joints[:,arm] = solved
    joints[:,fingers] = .04
    robot.write_joint_state_to_sim(joints,torch.zeros_like(joints))
    robot.set_joint_position_target(joints)
    env.sim.forward(); env.scene.update(0.)
    hp,hq,gp,gq = mdp.geometry(env)
    errors = (hp-gp).norm(dim=-1)
    angles = quat_error_magnitude(hq,gq)
    valid = (errors < .002) & (angles < .03)
    f = env.fixed_finish
    f.phase[:] = 1; f.phase_steps[:] = 0; f.position[:] = hp; f.rotation[:] = hq
    f.trigger[:] = 3  # Explicit diagnostic initialization, not alignment/deadline.
    f.handover_distance[:] = errors; f.handover_angle[:] = angles
    env.diagnostic_step = -1
    env._reset_idx = lambda ids: None
    initialization = []
    for i,s in enumerate(specs):
        initialization.append(dict(**s, diagnostic_base=diagnostic_roots[i],
            arm=solved[i].tolist(), position_error_m=float(errors[i]), angle_error_rad=float(angles[i]),
            valid=bool(valid[i]), object_state=cases[i]['object_state']))
    (out/'initialization.json').write_text(json.dumps(initialization,indent=2))
    print('HOLD_INITIALIZED', int(valid.sum()), '/', n, 'max_position_error',float(errors.max()),flush=True)
    if args.enable_cameras:
        import imageio.v2 as imageio
        from pxr import UsdGeom, UsdPhysics
        for prim in env.sim.stage.Traverse():
            if '/Table/collider_' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
                UsdGeom.Imageable(prim).CreatePurposeAttr('default')
        for _ in range(12): env.render()
        video = imageio.get_writer(out/'hold.mp4',fps=10,macro_block_size=16)
        frame=env.render().copy(); imageio.imwrite(out/'initial-frame.png',frame); video.append_data(frame)
    trace = []
    fatal = [None]*n
    close_anchor = torch.zeros(n,3,device=env.device)
    anchored = torch.zeros(n,dtype=torch.bool,device=env.device)
    ever_success = torch.zeros_like(anchored)
    initial_object = obj.data.root_pos_w.clone()
    total_ticks = sum(f.ticks(k) for k in ('settle_seconds','close_seconds','lift_seconds','hold_seconds'))
    for step in range(total_ticks):
        with torch.no_grad(): env.step(torch.zeros(n,10,device=env.device))
        d = env.diagnostics
        hp,hq,_,_ = mdp.geometry(env)
        relative = quat_apply_inverse(hq,obj.data.root_pos_w-hp)
        newly = (f.phase >= 3) & ~anchored
        close_anchor[newly] = relative[newly]; anchored |= newly
        drift = (relative-close_anchor).norm(dim=-1)
        ever_success |= d['success']
        for i in range(n):
            pairs=[]
            if bool(d['collision'][i]):
                for body,key in zip(BODIES,env.forbidden_sensors):
                    sensor=env.scene[key]
                    forces=sensor.data.force_matrix_w_history[i].norm(dim=-1).amax(dim=(0,1))
                    pairs.extend(dict(body=body,target=t,force_n=float(v)) for t,v in zip(sensor.cfg.filter_prim_paths_expr,forces) if v>.1)
            if fatal[i] is None:
                reason=next((k for k in ('collision','dropped','out_of_bounds') if bool(d[k][i])),None)
                if reason: fatal[i]=dict(reason=reason,seconds=(step+1)*env.step_dt,pairs=pairs)
            trace.append(dict(case=i,seconds=(step+1)*env.step_dt,phase=int(f.phase[i]),
                finger_force_n=d['finger_force'][i].tolist(),finger_effort=robot.data.applied_torque[i,fingers].tolist(),
                width_m=float(d['gripper_width'][i]),contact=bool(d['contact'][i]),
                supported=bool(d['supported'][i]),height_m=float(d['height'][i]),
                success=bool(d['success'][i]),collision=bool(d['collision'][i]),
                slip_m=float(d['slip'][i]),hold_seconds=float(env.grasp_state['hold'][i]),
                drift_from_lift_start_m=float(drift[i]) if anchored[i] else None,
                object_shift_xy_m=float((obj.data.root_pos_w[i,:2]-initial_object[i,:2]).norm()),
                hand_position=(hp[i]-env.scene.env_origins[i]).tolist(),
                object_position=(obj.data.root_pos_w[i]-env.scene.env_origins[i]).tolist()))
        if video is not None and (step+1)%2==0: video.append_data(env.render().copy())
    results=[]
    for i,s in enumerate(specs):
        rows=[r for r in trace if r['case']==i]; last=rows[-1]
        hold=[r for r in rows if r['phase']==4]; lift=[r for r in rows if r['phase']>=3]
        reason='initialization_failed' if not valid[i] else fatal[i]['reason'] if fatal[i] else 'success' if last['success'] else 'attempt_failed'
        results.append(dict(**s,reason=reason,first_fatal=fatal[i],ever_success=bool(ever_success[i]),
            final=last,maximum_lift_m=max(r['height_m'] for r in rows),
            max_drift_from_lift_start_m=max(r['drift_from_lift_start_m'] for r in lift),
            hold_double_contact_fraction=sum(r['contact'] for r in hold)/len(hold),
            hold_mean_finger_force_n=[sum(r['finger_force_n'][j] for r in hold)/len(hold) for j in range(2)],
            hold_min_finger_force_n=[min(r['finger_force_n'][j] for r in hold) for j in range(2)],
            hold_width_range_m=[min(r['width_m'] for r in hold),max(r['width_m'] for r in hold)]))
    summary=dict(status='complete',config=args.config,config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        diagnostic_joint_and_base_initialization=True,physical_parameters_changed=False,
        limitation='Known relative grasp geometry translated to 10 left-front positions; does not test approach, arbitrary base reachability or broad robustness.',
        simulated_seconds=total_ticks*env.step_dt,wall_seconds=time.perf_counter()-started,
        initialized_within_tolerance=int(valid.sum()),counts=dict(collections.Counter(r['reason'] for r in results)),
        per_reference={ref:dict(collections.Counter(r['reason'] for r in results if r['reference']==ref)) for ref in sorted({r['reference'] for r in results})},
        actuator_configuration=str(cfg.scene.robot.actuators['fingers']),results=results)
    (out/'trace.json').write_text(json.dumps(trace))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('CUP_HOLD_DONE',summary['counts'],flush=True)
except BaseException:
    traceback.print_exc()
    raise
finally:
    if video is not None: video.close()
    env.close(); app.close()
