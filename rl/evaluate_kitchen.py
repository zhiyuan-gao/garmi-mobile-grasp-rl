"""Paired kitchen evaluation/recording with the same configured_env as training."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--cases')
p.add_argument('--case-index', type=int)
p.add_argument('--num_envs', type=int, default=240)
p.add_argument('--seed', type=int, default=20261018)
p.add_argument('--out', required=True)
p.add_argument('--camera-eye', type=float, nargs=3, help='Optional recording-only camera position')
p.add_argument('--camera-lookat', type=float, nargs=3, help='Optional recording-only camera target')
p.add_argument('--save-rollouts', action='store_true', help='Save raw states/actions for diagnosing closed-loop drift')
p.add_argument('--dagger-queries', action='store_true', help='Record learner states for expert labels; NEVER execute expert actions')
p.add_argument('--dagger-probability', type=float, default=.5)
p.add_argument('--dagger-min-step', type=int, default=5)
p.add_argument('--dagger-max-step', type=int, default=18)
p.add_argument('--dagger-stop-after-queries',action='store_true',help='Interface audit only: end collection after query window, not a success-rate evaluation')
p.add_argument('--dagger-stochastic',action='store_true',help='Collect states using the PPO policy distribution instead of its deterministic mean')
p.add_argument('--giskard-stages',action='store_true',help='Record persistent teacher approach stage; never changes learner actions')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.dagger_queries and not args.cases:
    p.error('Expert-label collection requires explicit training cases; frozen evaluation cases cannot be used')
if args.dagger_stop_after_queries and not args.dagger_queries:
    p.error('--dagger-stop-after-queries requires --dagger-queries')
if args.dagger_stochastic and not args.dagger_queries:
    p.error('--dagger-stochastic requires --dagger-queries')
if args.giskard_stages and not args.dagger_queries:
    p.error('--giskard-stages requires --dagger-queries')
app = AppLauncher(args).app
import collections, copy, hashlib, json, math, time
import numpy as np
import torch, yaml
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner
from garmi_rl.config import BODIES, configured_env
from garmi_rl.giskard_io import advance_stages

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
profile = yaml.safe_load(Path(args.config).read_text())
from garmi_rl.joint_delta import ACTION_SPACE, is_joint_delta
checkpoint_metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('action_contract', {})
if is_joint_delta(profile) != (checkpoint_metadata.get('name') == ACTION_SPACE):
    raise ValueError('Checkpoint and environment action semantics differ')
case_path = Path(args.cases or profile['evaluation']['cases'])
case_hash = hashlib.sha256(case_path.read_bytes()).hexdigest()
if not args.cases and case_hash != profile['evaluation']['cases_sha256']:
    raise ValueError('Frozen evaluation cases changed')
cases = json.loads(case_path.read_text())
if args.case_index is not None:
    cases = [cases[args.case_index]]
count = min(args.num_envs, len(cases))
if args.enable_cameras and (len(cases) != 1 or count != 1):
    raise ValueError('Recording requires a single frozen case')
cfg, task = configured_env(profile, count, args.seed)
if args.enable_cameras:
    cfg.viewer.resolution = (960, 640)
    cfg.sim.render_interval = cfg.decimation * 2
    if args.camera_eye is not None:
        cfg.viewer.eye = tuple(args.camera_eye)
    if args.camera_lookat is not None:
        cfg.viewer.lookat = tuple(args.camera_lookat)
env = ManagerBasedRLEnv(cfg, render_mode='rgb_array' if args.enable_cameras else None)
env.forbidden_sensors = cfg.forbidden_sensors
wrapped = SkrlVecEnvWrapper(env)
profile['agent']['experiment'].update(directory=str(out), experiment_name='replay', write_interval=0)
agent = Runner(wrapped, profile).agent
agent.load(args.checkpoint)
agent.set_running_mode('eval')
if args.dagger_queries:
    torch.manual_seed(args.seed)
if args.enable_cameras:
    from pxr import UsdGeom, UsdPhysics
    for prim in env.sim.stage.Traverse():
        if '/Table/collider_' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.inherited)
            UsdGeom.Imageable(prim).CreatePurposeAttr('default')
original_reset = env._reset_idx
results, actual_cases, step_times = [], [], []
start = time.perf_counter()
max_steps = math.ceil(cfg.episode_length_s / env.step_dt) + 2
frames = []
queries = []
query_rng = np.random.default_rng(args.seed)
arm_ids = env.scene['robot'].find_joints('right_fr3_joint[1-7]')[0]
if args.dagger_queries:
    frozen = json.loads(Path(profile['evaluation']['cases']).read_text())
    frozen_keys = {tuple(c['robot_state']+c['object_state']) for c in frozen}
    assert not any(tuple(c['robot_state']+c['object_state']) in frozen_keys for c in cases)

def summarize(status):
    value = dict(status=status, independent_process_load=True, seed=args.seed,
        checkpoint=args.checkpoint, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        evaluation_cases_sha256=case_hash, source_case_index=args.case_index,
        num_envs=count, episodes=len(cases), completed=len(results), results=results,
        initial_cases=actual_cases, initial_states=[c['robot_state'] for c in actual_cases],
        task=task, success_rate=sum(r['reason']=='success' for r in results)/len(cases),
        counts=dict(collections.Counter(r['reason'] for r in results)),
        wall_seconds=time.perf_counter()-start,
        mean_step_ms=float(np.mean(step_times)*1000) if step_times else None,
        phase_seen=sorted({phase for r in results for phase in r['phases_seen']}))
    if args.enable_cameras:
        value['camera'] = dict(eye=list(cfg.viewer.eye), lookat=list(cfg.viewer.lookat),
                               resolution=list(cfg.viewer.resolution))
    if args.dagger_queries:
        value['dagger'] = dict(queries=len(queries),expert_actions_executed=0,
            query_probability=args.dagger_probability,query_min_step=args.dagger_min_step,
            query_max_step=args.dagger_max_step,successful_episode_filter=False,
            policy_mode='sampled PPO actions' if args.dagger_stochastic else 'deterministic mean for interface audit',
            note='Only label-query collection; no PPO training occurs in this evaluator')
        if args.dagger_stop_after_queries:
            value['success_rate']=None
            value['grasp_success_evaluation']=False
        (out/'dagger-queries.json').write_text(json.dumps(queries,indent=2,allow_nan=False))
    for key in ('reference','region','depth'):
        value['per_'+key] = {}
        for name in sorted({r[key] for r in results}):
            rows=[r for r in results if r[key]==name]
            value['per_'+key][name]=dict(episodes=len(rows),counts=dict(collections.Counter(r['reason'] for r in rows)),
                success_rate=sum(r['reason']=='success' for r in rows)/len(rows))
    temp=out/'summary.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(out/'summary.json')
    return value

with (out/'trace.jsonl').open('w') as trace:
    for batch in range(0,len(cases),count):
        chunk=cases[batch:batch+count]; n=len(chunk)
        padded=copy.deepcopy(chunk)+[copy.deepcopy(chunk[-1]) for _ in range(count-n)]
        env._reset_idx=original_reset
        env.reset()
        env.grasp_bank.restore(padded)
        actual=env.grasp_bank.cases()
        err=max(abs(a-b) for x,y in zip(actual,padded) for key in ('robot_state','object_state') for a,b in zip(x[key],y[key]))
        assert err < 2.e-5, err
        actual_cases += actual[:n]
        env._reset_idx=lambda ids: None
        obs=env.observation_manager.compute()['policy']
        if args.save_rollouts:
            recorded_obs=[obs.cpu().numpy().copy()]; recorded_actions=[]; recorded_ended=[]
        active=torch.arange(count,device=env.device)<n
        maxheight=torch.zeros(count,device=env.device)
        phases=torch.zeros((count,5),dtype=torch.bool,device=env.device)
        refs=env.grasp_bank.reference_ids.clone()
        initialbase=env.scene['robot'].data.root_pos_w.clone()
        travel=torch.zeros(count,device=env.device)
        batch_results=[None]*n
        teacher_stage=np.zeros(count,dtype=bool)
        if args.enable_cameras:
            for _ in range(12):env.render()
            frames.append(env.render().copy())
        for step in range(max_steps):
            pending_queries = {}
            if args.giskard_stages:
                teacher_stage=advance_stages(teacher_stage,obs.cpu().numpy())
            if args.dagger_queries and args.dagger_min_step < step < args.dagger_max_step:
                robot = env.scene['robot']
                selected = active & (env.fixed_finish.phase == 0) & torch.as_tensor(
                    query_rng.random(count)<args.dagger_probability,device=env.device)
                selected_ids=selected.nonzero().flatten()
                roots=robot.data.root_link_state_w[selected_ids].clone();roots[:,:3]-=env.scene.env_origins[selected_ids]
                objects=env.scene['object'].data.root_state_w[selected_ids].clone();objects[:,:3]-=env.scene.env_origins[selected_ids]
                captured=dict(robot_state=roots.tolist(),object_state=objects.tolist(),
                    arm_position=robot.data.joint_pos[selected_ids][:,arm_ids].tolist(),
                    arm_velocity=robot.data.joint_vel[selected_ids][:,arm_ids].tolist(),
                    base_servo_velocity=env.action_manager.get_term('base').velocity[selected_ids].tolist(),
                    last_action=env.action_manager.action[selected_ids].tolist(),observation=obs[selected_ids].tolist(),
                    elapsed_seconds=(env.fixed_finish.phase_steps[selected_ids]*env.step_dt).tolist(),
                    aligned_steps=env.fixed_finish.aligned_steps[selected_ids].tolist(),
                    episode_steps=env.episode_length_buf[selected_ids].tolist())
                if is_joint_delta(profile) and selected_ids.numel():
                    from garmi_rl.expert_actions import PARK
                    targets = env.action_manager.get_term('arm').processed_actions[selected_ids].cpu().numpy()
                    previous = np.array(captured['last_action'])
                    captured['policy_last_action'] = captured['last_action']
                    captured['last_action'] = np.concatenate((previous[:,:3], (targets-PARK)/1.5),axis=1).tolist()
                for row_index,i in enumerate(selected_ids.tolist()):
                    initial=padded[i]
                    pending_queries[i]=dict(id=f"{out.name}-c{batch+i:04d}-s{step:04d}",
                        reference=initial['reference'],initial_case=initial,case=batch+i,step=step,
                        snapshot=dict(object=initial['object'],reference=initial['reference'],
                            **{key:values[row_index] for key,values in captured.items()}),
                        approach_seconds=cfg.fixed_finish['approach_seconds'],step_dt=env.step_dt,
                        movement_bounds=profile['garmi_task']['kitchen']['movement_bounds_task_xy'],
                        learner_checkpoint=args.checkpoint)
                    if args.giskard_stages:
                        pending_queries[i]['approach_stage']='approach_grasp' if teacher_stage[i] else 'pregrasp'
            with torch.no_grad():
                action,_,extra=agent.act(obs,timestep=step,timesteps=max_steps)
                action=action if args.dagger_stochastic else extra.get('mean_actions',action)
                torch.cuda.synchronize();tick=time.perf_counter()
                obs,reward,done,truncated,_=wrapped.step(action)
                torch.cuda.synchronize();step_times.append(time.perf_counter()-tick)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all() and torch.isfinite(action).all()
            if args.save_rollouts:
                recorded_obs.append(obs.cpu().numpy().copy())
                recorded_actions.append(action.cpu().numpy().copy())
                recorded_ended.append((done|truncated).flatten().cpu().numpy().copy())
            assert torch.equal(refs,env.grasp_bank.reference_ids)
            d=env.diagnostics
            for i,query in pending_queries.items():
                if int(env.fixed_finish.phase[i]) != 0:
                    continue  # Fixed finish overrides policy actions at handover.
                query.update(learner_action=action[i].tolist(),next_observation=obs[i].tolist(),
                    terminal=bool((done|truncated).flatten()[i]))
                queries.append(query)
            ids=active.nonzero().flatten()
            phases[ids,d['phase'][ids]]=True
            maxheight[active]=torch.maximum(maxheight[active],d['height'][active])
            travel[active]=torch.maximum(travel[active],(d['base_position'][:,:2]-initialbase[:,:2]).norm(dim=-1)[active])
            ended=(done|truncated).flatten() & active
            for i in ended.nonzero().flatten().tolist():
                reason=next((key for key in ('collision','dropped','out_of_bounds','success','attempt_failed') if d[key][i]),'timeout')
                obj=padded[i]['object_state'];mapxy=[.3-obj[1],6.35+obj[0]]
                row=dict(case=batch+i,object=padded[i]['object'],reference=padded[i]['reference'],reason=reason,
                    seconds=(step+1)*env.step_dt,steps=step+1,region='right' if mapxy[0]>1 else 'left',
                    depth='front' if mapxy[1]<7.363333 else 'rear' if mapxy[1]>7.496667 else 'middle',
                    object_map_xy=mapxy, final_phase=int(d['phase'][i]), phases_seen=phases[i].nonzero().flatten().tolist(),
                    handover_trigger=int(d['handover_trigger'][i]),handover_distance=float(d['handover_distance'][i]),
                    handover_angle=float(d['handover_angle'][i]),height=float(d['height'][i]),
                    max_height=float(maxheight[i]),contact=bool(d['contact'][i]),slip=float(d['slip'][i]),
                    supported=bool(d['supported'][i]),hold_seconds=float(env.grasp_state['hold'][i]),
                    distance=float(d['distance'][i]),max_base_displacement=float(travel[i]))
                if reason=='success':
                    assert row['height']>=.08 and row['contact'] and not row['supported']
                    assert row['slip']<.02 and row['hold_seconds']>=2.
                if reason=='collision':
                    targets=[]
                    for body,key in zip(BODIES,env.forbidden_sensors):
                        sensor=env.scene[key]
                        force=sensor.data.force_matrix_w_history[i].norm(dim=-1).amax(dim=(0,1))
                        targets += [dict(body=body,target=target,force=float(f)) for target,f in zip(sensor.cfg.filter_prim_paths_expr,force) if f>.1]
                    row['contact_targets']=targets
                batch_results[i]=row
            trace.write(json.dumps(dict(batch=batch,step=step,active=int(active.sum()),ended=int(ended.sum()),
                mean_reward=float(reward.flatten()[active].mean()),
                **(dict(action=action.tolist(),distance=d['distance'].tolist(),phase=d['phase'].tolist(),height=d['height'].tolist()) if args.enable_cameras else {})))+'\n')
            active &= ~ended
            if args.enable_cameras and ((step+1)%2==0 or not active.any()):frames.append(env.render().copy())
            if not active.any():break
            if args.dagger_stop_after_queries and step+1>=args.dagger_max_step:
                for i in active.nonzero().flatten().tolist():
                    batch_results[i]=dict(case=batch+i,object=padded[i]['object'],reference=padded[i]['reference'],
                        reason='collection_window_end',region=padded[i]['region'],depth=padded[i]['depth'],
                        phases_seen=phases[i].nonzero().flatten().tolist(),steps=step+1)
                break
        assert all(r is not None for r in batch_results),'Unfinished evaluation episode'
        if args.save_rollouts:
            np.savez_compressed(out/f'rollout_{batch:04d}.npz', observations=np.asarray(recorded_obs),
                actions=np.asarray(recorded_actions), ended=np.asarray(recorded_ended))
        results += batch_results
        value=summarize('complete' if len(results)==len(cases) else 'running')
        print('EVALUATION_BATCH',len(results),'/',len(cases),value['counts'],flush=True)
if frames:
    import imageio.v2 as imageio
    imageio.imwrite(out/'initial-frame.png',frames[0])
    imageio.imwrite(out/'final-frame.png',frames[-1])
    imageio.mimsave(out/'replay.mp4',frames,fps=10,macro_block_size=16)
print('EVALUATION_DONE',json.dumps({k:value[k] for k in ('success_rate','counts','phase_seen','wall_seconds')}),flush=True)
env.close();app.close()
