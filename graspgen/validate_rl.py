"""Kitchen grasp diagnostics using the RL controller; never collect expert data.

IK initializes the robot directly at the reference. Only the existing physical
settle/close/lift/hold controller runs. This cannot validate an approach trajectory.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(__doc__)
p.add_argument('--config', required=True)
p.add_argument('--out', required=True)
p.add_argument('--num_envs', type=int, default=80)
p.add_argument('--object-collision-mesh', help='Additional cooked boundary for static object endpoint checks.')
p.add_argument('--cases', help='Explicit diagnostic cases with prescribed endpoint_base_pose and endpoint_arm_seed.')
p.add_argument('--placement-cases', help='Explicit object placements using the normal 54-choice endpoint IK search; mutually exclusive with --cases.')
p.add_argument('--seed', type=int, default=2026101821)
p.add_argument('--trace-phases',action='store_true',help='Record close/lift/hold geometry to distinguish object slip from hand tracking error.')
p.add_argument('--audit-wrist-contact',action='store_true',help='Record wrist contact targets and a counterfactual allowing only wrist/Object contact; original pass/fail remains unchanged.')
p.add_argument('--diagnostic-finger-stiffness',type=float,
    help='Isolated sensitivity experiment only; override is recorded and cannot be exported as default-profile success evidence.')
p.add_argument('--diagnostic-finger-friction',type=float,
    help='Isolated sensitivity experiment: set static/dynamic friction only on both finger bodies; record override and runtime readback.')
p.add_argument('--compare-grasp-center-slip',action='store_true',
    help='Diagnostic counterfactual only: compare slip at the GraspGen TCP against existing object-origin slip. Does not change pass/fail.')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert not (args.cases and args.placement_cases)
app = AppLauncher(args).app

import collections
import hashlib
import itertools
import json
import time
import traceback
import numpy as np
import torch
import trimesh
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms, quat_error_magnitude, quat_apply_inverse, quat_apply
from scipy.spatial.transform import Rotation
from garmi_rl.config import configured_env, PARK
from garmi_rl.diagnostic_ik import DiagnosticIK
from garmi_rl.kitchen import initial_collection_cases
from garmi_rl import mdp
from rl_geometry import RobotCollision


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def make_cases(mask, name):
    cases=[]
    missing=[]
    for j, ref in enumerate(mask['reference_ids']):
        for region in ('left','right'):
            pool=[c for c in mask['cells'] if c['region']==region and j in c['allowed_reference_indices']]
            if not pool:
                missing.append(dict(reference=ref,region=region));continue
            x_target=.16 if region=='left' else 1.30
            for depth,y_target in (('front',7.27),('rear',7.55)):
                cell=min(pool,key=lambda c: np.linalg.norm(np.mean(c['bounds_map_xy'],axis=0)-[x_target,y_target]))
                x,y=map(float,np.mean(cell['bounds_map_xy'],axis=0))
                cases.append(dict(object=name,reference=ref,region=region,depth=depth,
                    object_state=[y-6.35,.3-x,mask['bowl_origin_z_m'],2**-.5,0,0,-2**-.5,0,0,0,0,0,0],
                    robot_state=[0,.05,.0259,1,0,0,0,0,0,0,0,0,0]))
    return cases,missing


out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
profile=yaml.safe_load(Path(args.config).read_text())
cfg,_=configured_env(profile,args.num_envs,args.seed)
diagnostic_overrides={}
if args.diagnostic_finger_stiffness is not None:
    assert args.diagnostic_finger_stiffness>0
    diagnostic_overrides['finger_stiffness']=args.diagnostic_finger_stiffness
    cfg.scene.robot.actuators['fingers'].stiffness=args.diagnostic_finger_stiffness
env=ManagerBasedRLEnv(cfg)
env.forbidden_sensors=cfg.forbidden_sensors
started=time.perf_counter()
results=[]
try:
    env.reset()
    if args.diagnostic_finger_friction is not None:
        assert args.diagnostic_finger_friction>=0
        robot=env.scene['robot']
        counts=[robot._physics_sim_view.create_rigid_body_view(path).max_shapes
                for path in robot.root_physx_view.link_paths[0]]
        assert sum(counts)==robot.root_physx_view.max_shapes
        before=robot.root_physx_view.get_material_properties().clone()
        modified=before.clone();indices=[]
        bodies=robot.find_bodies('right_fr3_.*finger')[0]
        assert len(bodies)==2
        for body in bodies:
            start=sum(counts[:body]);indices.extend(range(start,start+counts[body]))
        assert len(indices)==8
        modified[:,indices,:2]=args.diagnostic_finger_friction
        robot.root_physx_view.set_material_properties(modified,torch.arange(env.num_envs,dtype=torch.int32,device='cpu'))
        actual=robot.root_physx_view.get_material_properties()
        assert torch.allclose(actual,modified)
        diagnostic_overrides['finger_friction']=dict(static=args.diagnostic_finger_friction,
            dynamic=args.diagnostic_finger_friction,shape_indices=indices,
            original_values=torch.unique(before[:,indices,:].reshape(-1,3),dim=0).tolist(),
            runtime_readback=torch.unique(actual[:,indices,:].reshape(-1,3),dim=0).tolist(),
            other_shapes_unchanged=True)
        save(out/'diagnostic-materials.json',diagnostic_overrides)
    bank=env.grasp_bank
    name=bank.objects[0]['name']
    mask=json.loads(Path(profile['garmi_task']['kitchen']['mask']).read_text())
    case_path=args.cases or args.placement_cases
    cases,missing=(json.loads(Path(case_path).read_text()),[]) if case_path else make_cases(mask,name)
    assert cases and all(c['object']==name and c['reference'] in mask['reference_ids'] for c in cases)
    if args.cases:
        assert len({c['id'] for c in cases})==len(cases)
        assert all(len(c['endpoint_base_pose'])==7 and len(c['endpoint_arm_seed'])==7 for c in cases)
    save(out/'cases.json',cases)
    # Exercise the normal reset + serialization + restore path, before any diagnostic initialization.
    max_roundtrip=0.
    for _ in range(10):
        env.reset();reset_cases,error=initial_collection_cases(bank);bank.restore(reset_cases)
        max_roundtrip=max(max_roundtrip,error)
    save(out/'reset-check.json',dict(status='passed',cases=10*args.num_envs,maximum_roundtrip_error_m=max_roundtrip))
    mesh=trimesh.load(bank.objects[0]['mesh'],force='mesh',process=False)
    if args.object_collision_mesh:
        mesh=trimesh.util.concatenate([mesh,trimesh.load(args.object_collision_mesh,force='mesh',process=False)])
    opening_m=cfg.fixed_finish.get('gripper_open_width_m',.08)
    checkers={}
    solver=DiagnosticIK(env.device)
    robot=env.scene['robot'];arm=robot.find_joints('right_fr3_joint[1-7]')[0]
    original_reset=env._reset_idx
    # Search endpoint configurations, not motions. These arrays are never expert demonstrations.
    choices=list(itertools.product((.08,.20,.32),(.15,.35,.55),(-.35,0.,.35),(0,1)))
    for batch,start in enumerate(range(0,len(cases),args.num_envs)):
        chunk=cases[start:start+args.num_envs];n=len(chunk)
        padded=chunk+[chunk[-1]]*(args.num_envs-n)
        env._reset_idx=original_reset;env.reset();bank.restore(padded)
        _,_,gp,gq=mdp.geometry(env)
        gp_local=gp-env.scene.env_origins
        nc=1 if args.cases else len(choices)
        bases=torch.zeros((args.num_envs,nc,7),device=env.device)
        if args.cases:
            bases[:,0]=torch.tensor([c['endpoint_base_pose'] for c in padded],device=env.device)
        else:
            for k,(x,dy,yaw,seed) in enumerate(choices):
                bases[:,k,0]=x;bases[:,k,1]=env.scene['object'].data.root_pos_w[:,1]-env.scene.env_origins[:,1]+dy
                bases[:,k,2]=.0259;bases[:,k,3]=np.cos(yaw/2);bases[:,k,6]=np.sin(yaw/2)
        b=bases.reshape(-1,7)
        target_p,target_q=subtract_frame_transforms(b[:,:3],b[:,3:7],
            gp_local[:,None,:].expand(-1,nc,-1).reshape(-1,3),gq[:,None,:].expand(-1,nc,-1).reshape(-1,4))
        if args.cases:
            initial=torch.tensor([c['endpoint_arm_seed'] for c in padded],device=env.device)
        else:
            initial=torch.tensor(PARK,device=env.device).expand(len(b),-1).clone()
            initial[1::2,0]=.7;initial[1::2,2]=-.5
        limits=robot.data.soft_joint_pos_limits[:,arm][:,None].expand(-1,nc,-1,-1).reshape(-1,7,2)
        q=solver.solve(target_p,target_q,initial,limits,iterations=240)
        if args.cases:
            # Nominal repeats reproduce saved joints exactly; perturbations use
            # local IK from that seed, with no search for an alternative base.
            nominal=torch.tensor([c['test_kind']=='nominal' for c in padded],device=env.device)
            q=torch.where(nominal[:,None],initial,q)
        fp,fq,_=solver.forward(q)
        pe=(fp-target_p).norm(dim=1);ae=quat_error_magnitude(fq,target_q)
        cost=(pe+.1*ae).reshape(args.num_envs,nc).cpu().numpy()
        qn=q.reshape(args.num_envs,nc,7).cpu().numpy();bn=bases.cpu().numpy()
        pn=pe.reshape(args.num_envs,nc).cpu().numpy();an=ae.reshape(args.num_envs,nc).cpu().numpy()
        picked=[];metadata=[]
        for i,case in enumerate(padded):
            case_opening=float(env.grasp_open_width[i])
            opening_key=round(case_opening,6)
            if opening_key not in checkers:
                checkers[opening_key]=RobotCollision(mesh,opening_m=opening_key)
            checker=checkers[opening_key]
            order=np.argsort(cost[i]);selected=int(order[0]);rejections=collections.Counter();clear=False
            op=np.eye(4);op[:3,3]=case['object_state'][:3]
            op[:3,:3]=Rotation.from_quat(np.array(case['object_state'])[ [4,5,6,3] ]).as_matrix()
            for k in order:
                if pn[i,k]>.005 or an[i,k]>.05:continue
                bp=np.eye(4);bp[:3,3]=bn[i,k,:3]
                bp[:3,:3]=Rotation.from_quat(bn[i,k,[4,5,6,3]]).as_matrix()
                reason=checker.check(bp,qn[i,k],op)
                if reason:rejections[reason]+=1
                else:selected=int(k);clear=True;break
            picked.append(selected)
            metadata.append(dict(ik_position_error_m=float(pn[i,selected]),ik_angle_error_rad=float(an[i,selected]),
                endpoint_geometry_clear=clear,endpoint_rejections=dict(rejections),base_pose=bn[i,selected].tolist(),
                arm_position=qn[i,selected].tolist(),pregrasp_open_width_m=case_opening))
        idx=torch.tensor(picked,device=env.device);ar=torch.arange(args.num_envs,device=env.device)
        selected_q=q.reshape(args.num_envs,nc,7)[ar,idx]
        selected_b=bases[ar,idx].clone();selected_b[:,:3]+=env.scene.env_origins
        root=robot.data.default_root_state.clone();root[:,:7]=selected_b;root[:,7:]=0
        robot.write_root_pose_to_sim(root[:,:7]);robot.write_root_velocity_to_sim(root[:,7:])
        joint=robot.data.default_joint_pos.clone();joint[:,arm]=selected_q
        joint[:,robot.find_joints('right_fr3_finger_joint.*')[0]]=env.grasp_open_width[:,None]/2
        robot.write_joint_state_to_sim(joint,torch.zeros_like(joint));robot.set_joint_position_target(joint)
        env.sim.forward();env.scene.update(0.)
        hp,hq,goalp,goalq=mdp.geometry(env)
        actual_pe=(hp-goalp).norm(dim=1).cpu().numpy()
        actual_ae=quat_error_magnitude(hq,goalq).cpu().numpy()
        for i,m in enumerate(metadata):
            m['isaac_hand_error_m']=float(actual_pe[i]);m['isaac_hand_angle_error_rad']=float(actual_ae[i])
        finish=env.fixed_finish
        finish.phase[:]=1;finish.phase_steps[:]=0;finish.position[:]=hp;finish.rotation[:]=hq
        finish.target_position[:]=hp;finish.target_rotation[:]=hq
        finish.handover_distance[:]=(hp-goalp).norm(dim=1)
        finish.handover_angle[:]=quat_error_magnitude(hq,goalq)
        env._reset_idx=lambda ids:None
        max_height=torch.zeros(args.num_envs,device=env.device)
        max_hold=torch.zeros_like(max_height);max_collision=torch.zeros_like(max_height)
        two_contacts=torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
        success=two_contacts.clone();out_of_bounds=two_contacts.clone()
        # Diagnostic counters explain failures without altering the controller or
        # acceptance criteria. Count only the final hold phase separately.
        hold_ticks=torch.zeros(args.num_envs,device=env.device)
        hold_contact=hold_ticks.clone();hold_high=hold_ticks.clone()
        hold_slip=hold_ticks.clone();hold_supported=hold_ticks.clone()
        forbidden_by_sensor=torch.zeros((args.num_envs,len(env.forbidden_sensors)),device=env.device)
        if args.audit_wrist_contact:
            wrist_key='forbidden_8'  # BODIES[8] is right_fr3_link7 (wrist/palm).
            assert cfg.scene.forbidden_8.prim_path.endswith('/right_fr3_link7')
            wrist_paths=cfg.scene.forbidden_8.filter_prim_paths_expr
            wrist_object_indices=[i for i,path in enumerate(wrist_paths) if path.endswith('/Object')]
            assert len(wrist_object_indices)==1
            wrist_object_index=wrist_object_indices[0]
            wrist_other_indices=[i for i in range(len(wrist_paths)) if i!=wrist_object_index]
            wrist_targets_max=torch.zeros((args.num_envs,len(wrist_paths)),device=env.device)
            relaxed_hold=torch.zeros_like(max_hold);relaxed_max_hold=relaxed_hold.clone();relaxed_max_force=relaxed_hold.clone()
        first_collision_phase=torch.full((args.num_envs,),-1,device=env.device,dtype=torch.long)
        phase_traces=[[] for _ in range(n)]
        center_anchor=torch.zeros((args.num_envs,3),device=env.device)
        center_previous_contact=two_contacts.clone()
        center_hold=hold_ticks.clone();center_max_hold=hold_ticks.clone()
        center_slip=hold_ticks.clone()
        for step in range(round(8.5/env.step_dt)):
            env.step(torch.zeros((args.num_envs,10),device=env.device))
            d=mdp.diagnostics(env)
            max_height=torch.maximum(max_height,d['height']);max_hold=torch.maximum(max_hold,env.grasp_state['hold'])
            max_collision=torch.maximum(max_collision,d['forbidden_force'].amax(dim=1))
            two_contacts|=d['contact'];success|=d['success'];out_of_bounds|=d['out_of_bounds']
            holding=finish.phase==4
            hold_ticks+=holding;hold_contact+=holding & d['contact']
            hold_high+=holding & (d['height']>=.08)
            hold_slip+=holding & (d['slip']>=.02)
            hold_supported+=holding & d['supported']
            forbidden_by_sensor=torch.maximum(forbidden_by_sensor,d['forbidden_force'])
            if args.audit_wrist_contact:
                matrix=env.scene[wrist_key].data.force_matrix_w_history
                assert matrix.shape[-2]==len(wrist_paths), (matrix.shape,wrist_paths)
                by_target=matrix.norm(dim=-1).amax(dim=(1,2))
                wrist_targets_max=torch.maximum(wrist_targets_max,by_target)
                other_sensors=[i for i,k in enumerate(env.forbidden_sensors) if k!=wrist_key]
                other_force=torch.maximum(d['forbidden_force'][:,other_sensors].amax(dim=1),by_target[:,wrist_other_indices].amax(dim=1))
                relaxed_max_force=torch.maximum(relaxed_max_force,other_force)
                relaxed_held=d['contact']&(d['height']>=.08)&~d['supported']&(d['slip']<.02)&(other_force<=.1)
                relaxed_hold=torch.where(relaxed_held,relaxed_hold+env.step_dt,0.)
                relaxed_max_hold=torch.maximum(relaxed_max_hold,relaxed_hold)
            first=(first_collision_phase<0)&d['collision']
            first_collision_phase[first]=finish.phase[first]
            if args.compare_grasp_center_slip:
                chp,chq,cgp,cgq=mdp.geometry(env)
                tcp=torch.zeros_like(cgp);tcp[:,2]=.0934
                relative_center=quat_apply_inverse(chq,cgp+quat_apply(cgq,tcp)-chp)
                new_contact=d['contact']&~center_previous_contact
                center_anchor[new_contact]=relative_center[new_contact]
                center_slip=(relative_center-center_anchor).norm(dim=1)
                center_held=d['contact']&(d['height']>=.08)&~d['supported']&(center_slip<.02)&~d['collision']
                center_hold=torch.where(center_held,center_hold+env.step_dt,0.)
                center_max_hold=torch.maximum(center_max_hold,center_hold)
                center_previous_contact=d['contact'].clone()
            if args.trace_phases and step in (8,48,108,169):
                trace_hp,trace_hq,_,_=mdp.geometry(env)
                tracking_position=(trace_hp-finish.target_position).norm(dim=1)
                tracking_angle=quat_error_magnitude(trace_hq,finish.target_rotation)
                relative=quat_apply_inverse(trace_hq,env.scene['object'].data.root_pos_w-trace_hp)
                for i in range(n):
                    phase_traces[i].append(dict(time_s=(step+1)*env.step_dt,phase=int(finish.phase[i]),
                        slip_m=float(d['slip'][i]),height_m=float(d['height'][i]),
                        hand_tracking_position_error_m=float(tracking_position[i]),
                        hand_tracking_angle_error_rad=float(tracking_angle[i]),
                        object_origin_in_hand=relative[i].cpu().tolist(),
                        finger_force_n=d['finger_force'][i].cpu().tolist(),
                        gripper_width_m=float(d['gripper_width'][i])))
                    if args.compare_grasp_center_slip:
                        phase_traces[i][-1].update(grasp_center_slip_m=float(center_slip[i]),
                            object_grasp_angle_error_rad=float(d['angle'][i]))
        for i,case in enumerate(chunk):
            m=metadata[i]
            aligned=m['isaac_hand_error_m']<=.005 and m['isaac_hand_angle_error_rad']<=.05
            passed=bool(success[i]) and m['endpoint_geometry_clear'] and aligned and float(max_collision[i])<=.1 and not bool(out_of_bounds[i])
            reason=('ik_unreachable' if not aligned else 'endpoint_collision' if not m['endpoint_geometry_clear'] else
                    'forbidden_collision' if max_collision[i]>.1 else 'out_of_bounds' if out_of_bounds[i] else
                    'success' if passed else 'no_two_finger_contact' if not two_contacts[i] else 'lift_or_hold_failed')
            extra={k:case[k] for k in ('id','condition_id','test_kind','offset_task_xy_m') if k in case}
            results.append(dict(reference=case['reference'],region=case['region'],depth=case['depth'],**extra,
                passed=passed,reason=reason,**m,max_lift_m=float(max_height[i]),max_hold_s=float(max_hold[i]),
                max_forbidden_force_n=float(max_collision[i]),ever_two_finger_contact=bool(two_contacts[i]),
                hold_phase_diagnostics=dict(ticks=int(hold_ticks[i]),two_finger_contact_ticks=int(hold_contact[i]),
                    sufficient_height_ticks=int(hold_high[i]),excessive_slip_ticks=int(hold_slip[i]),
                    supported_ticks=int(hold_supported[i])),
                first_forbidden_collision_phase=int(first_collision_phase[i]),
                phase_trace=phase_traces[i],
                grasp_center_slip_comparison=(dict(max_hold_s=float(center_max_hold[i]),
                    hypothetical_passed=bool(center_max_hold[i]>=2.) and m['endpoint_geometry_clear'] and aligned
                        and float(max_collision[i])<=.1 and not bool(out_of_bounds[i]),
                    acceptance_unchanged=True) if args.compare_grasp_center_slip else None),
                maximum_forbidden_force_by_sensor_n=dict(zip(env.forbidden_sensors,forbidden_by_sensor[i].cpu().tolist()))))
            if args.audit_wrist_contact:
                results[-1]['wrist_contact_audit']=dict(
                    maximum_force_by_target_n=dict(zip(wrist_paths,wrist_targets_max[i].cpu().tolist())),
                    hypothetical_pass_allowing_wrist_object=bool(relaxed_max_hold[i]>=2.) and m['endpoint_geometry_clear'] and aligned
                        and float(relaxed_max_force[i])<=.1 and not bool(out_of_bounds[i]),
                    maximum_other_forbidden_force_n=float(relaxed_max_force[i]),max_hold_allowing_wrist_object_s=float(relaxed_max_hold[i]),
                    original_acceptance_unchanged=True)
        save(out/'progress.json',dict(completed=len(results),total=len(cases),results=results))
        print('RL_DIAGNOSTIC_BATCH',name,batch,collections.Counter(r['reason'] for r in results),flush=True)
    byref={ref:dict(attempts=sum(r['reference']==ref for r in results),
        successes=sum(r['reference']==ref and r['passed'] for r in results),
        reasons=dict(collections.Counter(r['reason'] for r in results if r['reference']==ref))) for ref in mask['reference_ids']}
    save(out/'summary.json',dict(status='completed',object=name,reference_count=len(mask['reference_ids']),case_count=len(results),
        tested_reference_count=len({r['reference'] for r in results}),seed=args.seed,
        prescribed_cases_sha256=hashlib.sha256(Path(args.cases).read_bytes()).hexdigest() if args.cases else None,
        placement_cases_sha256=hashlib.sha256(Path(args.placement_cases).read_bytes()).hexdigest() if args.placement_cases else None,
        references_with_success=sum(v['successes']>0 for v in byref.values()),
        counts=dict(collections.Counter(r['reason'] for r in results)),per_reference=byref,results=results,
        missing_placement_regions=missing,wall_seconds=time.perf_counter()-started,
        configuration_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        diagnostic_overrides=diagnostic_overrides,
        pregrasp_open_width_m=opening_m,
        contact_policy=cfg.contact_policy,
        additional_object_collision_mesh=args.object_collision_mesh,
        additional_object_collision_mesh_sha256=(hashlib.sha256(Path(args.object_collision_mesh).read_bytes()).hexdigest()
                                                if args.object_collision_mesh else None),
        expert_trajectories_generated=0,optimizer_updates=0,
        scope='Direct IK initialization, cooked full-robot static endpoint checks, RL settle/close/lift/hold controller and original success/contact/collision thresholds.',
        limitations=['No Giskard approach trajectories; awaiting user confirmation.',
                    ('Prescribed endpoint conditions, exact nominal joints and local IK for perturbations; deterministic repeats do not estimate real-world reliability.' if args.cases else
                     'Finite base/arm IK seeds and four placements per reference; failures are diagnostic, not proofs of global infeasibility.'),
                    f"Object static/dynamic friction={bank.objects[0].get('friction_assumption', 'see asset material')} is an uncalibrated simulation assumption; robot uses default static/dynamic friction=0.5 unless an explicit diagnostic override is recorded. Masses use YCB metadata."]))
    print('YCB_RL_DIAGNOSTIC_DONE',name,collections.Counter(r['reason'] for r in results),flush=True)
except BaseException:
    traceback.print_exc();app.set_setting('/app/fastShutdown',False);raise
finally:
    env.close();app.close()
