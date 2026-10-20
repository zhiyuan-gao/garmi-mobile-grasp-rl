"""Collect physical Giskard demonstrations from the unchanged random reset distribution."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--out', required=True)
p.add_argument('--num_envs', type=int, default=128)
p.add_argument('--workers', type=int, default=8)
p.add_argument('--success-per-reference', type=int, default=50)
p.add_argument('--validation-per-reference', type=int, default=20)
p.add_argument('--max-batches', type=int, default=24)
p.add_argument('--seed', type=int, default=20261040)
p.add_argument('--dart', action='store_true', help='Perturb expert motion, then replan; never imitate perturbations')
p.add_argument('--audit-cases', help='Execute every supplied case once, retaining failures; no success-quota gate')
p.add_argument('--focus-references', nargs='+', help='Collect quotas only for these references from the configured bank')
p.add_argument('--balance-collection', action='store_true', help='Assign underfilled allowed references to sampled positions')
p.add_argument('--resume', action='store_true', help='Keep saved successes and continue quotas with an explicitly new seed')
p.add_argument('--diagnostics', action='store_true', help='Save collision target pairs and before/after expert states for failures')
p.add_argument('--allow-parallel-resize', action='store_true', help='Explicitly allow local strict collection to resume with a different environment count')
p.add_argument('--local-sampling', help='Frozen local start specification; balances remaining split/reference quotas')
p.add_argument('--require-strict-success', action='store_true', help='Require strict handover AND actual close-start accuracy for dataset acceptance')
p.add_argument('--bootstrap-train', nargs='+', default=[], help='Existing successful train indices; preserve their split')
p.add_argument('--bootstrap-validation', nargs='+', default=[], help='Existing successful validation indices; preserve their split')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.allow_parallel_resize:
    assert args.resume and args.local_sampling and args.require_strict_success
app = AppLauncher(args).app

import collections
import hashlib
import json
import math
import time
import traceback
import numpy as np
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from garmi_rl.config import BODIES, configured_env
from garmi_rl.giskard_io import COMMAND_VERSION, GiskardClient, advance_stages, entering_finish, snapshot_queries
from garmi_rl.joint_delta import COMMAND_VERSION as DELTA_VERSION, is_joint_delta, encode_joint_target
from garmi_rl.augmentation import sample_hand_perturbation, approach_label_mask
from garmi_rl import mdp
from garmi_rl.collision_pairs import planner_collision_settings
from garmi_rl.expert_collection import LocalCaseSampler, excluded_cases, strict_success
from scipy.spatial.transform import Rotation

torch.set_num_threads(4)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=args.resume)
(out/'trajectories').mkdir(exist_ok=args.resume)
profile = yaml.safe_load(Path(args.config).read_text())
cfg, task = configured_env(profile, args.num_envs, args.seed)
action_dim = 10
joint_delta = is_joint_delta(profile)
if joint_delta:
    COMMAND_VERSION = DELTA_VERSION
env = ManagerBasedRLEnv(cfg)
env.forbidden_sensors = cfg.forbidden_sensors
original_reset = env._reset_idx
object_names = profile['garmi_task']['multi_object']['objects']
if len(object_names) != 1:
    raise ValueError('Giskard collection currently requires one object per run')
object_name = object_names[0]
references = profile['garmi_task']['multi_object']['reference_ids'][object_name]
if args.focus_references:
    assert set(args.focus_references) <= set(references)
    references = args.focus_references
audit_cases = json.loads(Path(args.audit_cases).read_text()) if args.audit_cases else None
if audit_cases is not None:
    assert len(audit_cases) and not args.dart and not args.balance_collection
if args.local_sampling:
    assert audit_cases is None and not args.balance_collection and args.require_strict_success
quotas = {'train': args.success_per_reference, 'validation': args.validation_per_reference}
accepted = {split: collections.Counter() for split in quotas}
demonstrations = {split: [] for split in quotas}
results, audits, batch_timings = [], [], []
failure_queries = []
started = time.perf_counter()
client = GiskardClient(out/'giskard-service.log', args.workers,
    manifest=profile['garmi_task']['multi_object']['manifest'] if object_name != 'bowl' else None,
    object_name=object_name, precision=profile.get('expert_precision'),
    contact_policy=profile['garmi_task'].get('contact_policy','fingers_only'),
    **planner_collision_settings(profile))
maximum_action_error = 0.
dart_events = []
dart_rng = np.random.default_rng(args.seed + 1)
# The official environment uses 150 physics ticks at 1 kHz per action.
dart_ticks = round(.15 / env.step_dt)
assert np.isclose(dart_ticks * env.step_dt, .15)
frozen = json.loads(Path(profile['evaluation']['cases']).read_text())
heldout_keys = {tuple(c['robot_state']+c['object_state']) for c in frozen}
local_spec = json.loads(Path(args.local_sampling).read_text()) if args.local_sampling else None
local_spec_sha256 = hashlib.sha256(Path(args.local_sampling).read_bytes()).hexdigest() if local_spec else None
bootstrap_sources = []
previous_seconds = 0.
start_batch = 0
collection_segments = []
maximum_reset_roundtrip_error = 0.
if args.resume:
    import shutil
    import re
    previous = json.loads((out/'summary.json').read_text())
    assert previous['status'] in ('failed', 'collecting') and audit_cases is None
    assert previous['configuration_sha256'] == hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    assert previous['num_envs'] == args.num_envs or args.allow_parallel_resize
    assert previous['quotas'] == quotas
    assert previous['quota_references'] == references and previous['balance_collection'] == args.balance_collection
    assert previous['command_version'] == COMMAND_VERSION
    assert bool(previous['augmentation']) == args.dart
    assert previous.get('local_sampling_sha256') == local_spec_sha256
    assert previous.get('require_strict_success', False) == args.require_strict_success
    assert not args.bootstrap_train and not args.bootstrap_validation
    bootstrap_sources = previous.get('bootstrap_sources', [])
    collection_segments = previous.get('collection_segments', [])
    if not collection_segments:
        collection_segments = [dict(seed=previous['seed'], seconds=previous['wall_seconds'], status=previous['status'])]
    for segment in collection_segments:
        segment.setdefault('num_envs', previous['num_envs'])
        segment.setdefault('workers', previous.get('workers'))
    assert args.seed not in {s['seed'] for s in collection_segments}, 'Resume needs a new independent sampling seed'
    archive = out/f'recovery-before-seed-{args.seed}'
    archive.mkdir()
    for name in ('summary.json','train-index.json','validation-index.json','command-audit.json','perturbations.json'):
        if (out/name).exists(): shutil.copy2(out/name, archive/name)
    results = previous['results']
    for split in quotas:
        demonstrations[split] = json.loads((out/f'{split}-index.json').read_text())['demonstrations']
        for row in demonstrations[split]:
            assert row['reason'] == 'success' and row['selected_for_dataset']
            assert hashlib.sha256(Path(row['file']).read_bytes()).hexdigest() == row['sha256']
        accepted[split] = collections.Counter(r['reference'] for r in demonstrations[split])
        assert dict(accepted[split]) == previous['accepted'][split]
        assert all(n <= quotas[split] for n in accepted[split].values())
    assert len({r['id'] for rows in demonstrations.values() for r in rows}) == sum(map(len,demonstrations.values()))
    audits = json.loads((out/'command-audit.json').read_text())
    if args.dart: dart_events = json.loads((out/'perturbations.json').read_text())
    maximum_action_error = previous['maximum_action_alignment_error']
    maximum_reset_roundtrip_error = previous.get('maximum_reset_roundtrip_error_m', 0.)
    previous_seconds = previous['wall_seconds']
    start_batch = 1 + max((int(re.search(r'-b(\d+)-c', r['id'])[1]) for r in results), default=-1)
elif args.bootstrap_train or args.bootstrap_validation:
    assert local_spec is not None and args.require_strict_success
    imported_ids, imported_states = {}, set()
    for split, paths in [('train', args.bootstrap_train), ('validation', args.bootstrap_validation)]:
        for source in paths:
            source_path = Path(source)
            manifest = json.loads(source_path.read_text())
            assert manifest['status'] == 'complete' and manifest['command_version'] == COMMAND_VERSION
            bootstrap_sources.append(dict(split=split, file=source, sha256=hashlib.sha256(source_path.read_bytes()).hexdigest()))
            for row in manifest['demonstrations']:
                assert row['split'] == split and row['initial_case']['object'] == object_name
                assert strict_success(row) and bool(row.get('augmentation')) == args.dart
                assert tuple(row['initial_case']['robot_state']+row['initial_case']['object_state']) not in heldout_keys
                if row['id'] in imported_ids:
                    assert imported_ids[row['id']] == (split, row['sha256'])
                    continue
                imported_ids[row['id']] = (split, row['sha256'])
                assert hashlib.sha256(Path(row['file']).read_bytes()).hexdigest() == row['sha256']
                if row['reference'] not in references or accepted[split][row['reference']] >= quotas[split]:
                    continue
                key = tuple(row['initial_case']['robot_state']+row['initial_case']['object_state'])
                assert key not in imported_states
                imported_states.add(key)
                demonstrations[split].append(dict(row, strict_success_verified=True))
                accepted[split][row['reference']] += 1
local_sampler = None
if local_spec is not None:
    assert local_spec['object'] == object_name and set(references) <= set(local_spec['references'])
    excluded = frozen + excluded_cases(local_spec)
    excluded += [r['initial_case'] for rows in demonstrations.values() for r in rows]
    excluded += [r['initial_case'] for r in results]
    local_sampler = LocalCaseSampler(local_spec, args.seed+2, excluded)
collection_segments.append(dict(seed=args.seed, start_batch=start_batch, status='running',
                               num_envs=args.num_envs, workers=args.workers))


def atomic(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def complete():
    if audit_cases is not None:
        return len(results) == len(audit_cases)
    return all(accepted[split][ref] >= quota for split, quota in quotas.items() for ref in references)


def save(status, **extra):
    value = dict(status=status, seed=args.seed, num_envs=args.num_envs, workers=args.workers, results=results,
        counts=dict(collections.Counter(r['reason'] for r in results)), accepted=accepted,
        quotas=quotas, wall_seconds=previous_seconds+time.perf_counter()-started, timing=client.timing(),
        task=task, expert_precision=profile.get('expert_precision'), command_version=COMMAND_VERSION, configuration_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        collection=('Frozen local XY windows around successful PARK-arm approach starts; split/reference balanced quotas'
                    if local_spec else 'random normal reset; no IK/path screening or teleport to grasp'),
        augmentation='ga_ddpg_dart' if args.dart else None,
        dart=dict(events=len(dart_events),translation_per_axis_m=.04,rotation_per_axis_rad=.2,
            probability=.5,logical_expert_steps=[6,12],logical_action_seconds=.15,
            execution='Giskard target for 0.15 s; normal episode clock continues',
            perturbation_actions_are_labels=False) if args.dart else None,
        selection='train/validation assignment before execution; first physical successes per reference',
        maximum_action_alignment_error=maximum_action_error, **extra)
    collection_segments[-1].update(seconds=time.perf_counter()-started, status=status)
    value.update(audit_cases=args.audit_cases, quota_references=references,
                 batch_timings_seconds=batch_timings, allow_parallel_resize=args.allow_parallel_resize,
                 local_sampling=args.local_sampling, local_sampling_sha256=local_spec_sha256,
                 require_strict_success=args.require_strict_success, bootstrap_sources=bootstrap_sources,
                 strict_successes=sum(strict_success(r) for r in results),
                 collection_segments=collection_segments, resumed=args.resume,
                 maximum_reset_roundtrip_error_m=maximum_reset_roundtrip_error,
        balance_collection=args.balance_collection,
                 selection=('Every supplied audit case, without success filtering' if audit_cases is not None
                            else 'First physical successes per reference; split assigned before execution'))
    atomic(out/'summary.json', value)
    for split, rows in demonstrations.items():
        atomic(out/f'{split}-index.json', dict(status=status, expert_backend='giskard',
            command_version=COMMAND_VERSION, demonstrations=rows, trajectories=len(rows),
            physical_success_verified=True))
    atomic(out/'command-audit.json', audits)
    if args.diagnostics:
        atomic(out/'failure-queries.json', failure_queries)
    if args.dart:
        atomic(out/'perturbations.json', dart_events)


try:
    batch_limit = math.ceil(len(audit_cases)/args.num_envs) if audit_cases is not None else start_batch+args.max_batches
    for batch in range(start_batch, batch_limit):
        if complete():
            break
        env._reset_idx = original_reset
        env.reset()
        cases = env.grasp_bank.cases()
        if args.balance_collection:
            from garmi_rl.kitchen import initial_collection_cases
            cases, roundtrip_error = initial_collection_cases(env.grasp_bank)
            maximum_reset_roundtrip_error = max(maximum_reset_roundtrip_error, roundtrip_error)
        splits = ['validation' if (batch*args.num_envs+i) % 4 == 0 else 'train' for i in range(args.num_envs)]
        valid_count = len(cases)
        if local_sampler is not None:
            import copy
            chunk = local_sampler.sample(args.num_envs, references, accepted, quotas)
            valid_count = len(chunk)
            assert valid_count, 'Unfilled quota without an available local reference'
            planned_splits = [c['split'] for c in chunk]
            env.grasp_bank.restore(copy.deepcopy(chunk) + [copy.deepcopy(chunk[-1]) for _ in range(args.num_envs-valid_count)])
            cases = env.grasp_bank.cases()
            splits = planned_splits + [planned_splits[-1]]*(args.num_envs-valid_count)
            for i in range(valid_count):
                error = max(abs(a-b) for field in ['robot_state','object_state'] for a,b in zip(chunk[i][field],cases[i][field]))
                maximum_reset_roundtrip_error = max(maximum_reset_roundtrip_error, error)
                assert error < 5.e-5, ('Local case restore drift', error)
                cases[i]['source_window'] = chunk[i]['source_window']
        elif audit_cases is not None:
            import copy
            chunk = audit_cases[batch*args.num_envs:(batch+1)*args.num_envs]
            valid_count = len(chunk)
            env.grasp_bank.restore(copy.deepcopy(chunk) + [copy.deepcopy(chunk[-1]) for _ in range(args.num_envs-valid_count)])
            cases = env.grasp_bank.cases()
        elif args.balance_collection:
            from garmi_rl.kitchen import ensure_filter
            position_filter = ensure_filter(env.grasp_bank)
            xy = [[.3-c['object_state'][1], 6.35+c['object_state'][0]] for c in cases]
            allowed = position_filter.allowed(xy, 0., env.grasp_bank.pools[0]).cpu().numpy()
            ids = position_filter.data['reference_ids']
            assigned = {s: collections.Counter(accepted[s]) for s in quotas}
            for i, case in enumerate(cases):
                split = splits[i]
                pool = [ref for j,ref in enumerate(ids) if allowed[i,j] and ref in references
                        and assigned[split][ref] < quotas[split]]
                if pool:
                    smallest = min(assigned[split][ref] for ref in pool)
                    pool = [ref for ref in pool if assigned[split][ref] == smallest]
                    case['reference'] = str(dart_rng.choice(pool))
                    assigned[split][case['reference']] += 1
            env.grasp_bank.restore(cases)
        for i, case in enumerate(cases):
            case['id'] = f'{out.name}-s{args.seed}-b{batch:03d}-c{i:04d}'
            case['object_map_xy'] = [.3-case['object_state'][1], 6.35+case['object_state'][0]]
            x, y = case['object_map_xy']
            case['region'] = 'right' if x > 1 else 'left'
            case['depth'] = 'front' if y < 7.363333 else 'rear' if y > 7.496667 else 'middle'
            assert tuple(case['robot_state']+case['object_state']) not in heldout_keys
        active = np.array([i < valid_count and (audit_cases is not None or
            (c['reference'] in references and accepted[splits[i]][c['reference']] < quotas[splits[i]]))
            for i, c in enumerate(cases)])
        if not active.any():
            continue
        env._reset_idx = lambda ids: None
        obs = env.observation_manager.compute()['policy']
        count = args.num_envs
        max_steps = math.ceil(cfg.episode_length_s/env.step_dt)+2
        xs = np.empty((max_steps+1, count, 64), np.float32)
        ys = np.zeros((max_steps, count, action_dim), np.float32)
        rewards = np.zeros((max_steps, count), np.float32)
        phases = np.zeros((max_steps+1, count), np.int8)
        bc = np.zeros((max_steps, count), bool)
        terminated = np.zeros_like(bc)
        truncateds = np.zeros_like(bc)
        teacher_stages = np.zeros((max_steps, count), bool)
        perturbation_mask = np.zeros((max_steps, count), bool)
        normal_ticks = np.zeros(count, int)
        last_dart_slot = np.full(count, -1, int)
        dart_remaining = np.zeros(count, int)
        dart_targets = np.zeros((count, 7))
        event_ids = np.full(count, -1, int)
        perturbation_counts = np.zeros(count, int)
        stage = np.zeros(count, bool)
        last_expert_diagnostics = [None] * count
        xs[0] = obs.cpu().numpy()
        phases[0] = env.fixed_finish.phase.cpu().numpy()
        batch_start = time.perf_counter()
        for step in range(max_steps):
            stage = advance_stages(stage, xs[step])
            phase = env.fixed_finish.phase.cpu().numpy()
            selected = np.flatnonzero(active & (phase == 0) & ~entering_finish(env, xs[step])).tolist()
            requests = snapshot_queries(env, xs[step], cases, selected, stage, f'b{batch:03d}', step)
            if args.dart:
                hp,hq,_,_=mdp.geometry(env)
                positions=(hp-env.scene.env_origins).cpu().numpy();rotations=hq.cpu().numpy()
                for i in selected:
                    slot=normal_ticks[i]//dart_ticks
                    if dart_remaining[i]==0 and normal_ticks[i]%dart_ticks==0 and 5<slot<13 and last_dart_slot[i]!=slot:
                        last_dart_slot[i]=slot
                        if dart_rng.random()<.5:
                            noise=sample_hand_perturbation(positions[i],rotations[i],dart_rng)
                            dart_targets[i]=noise['pose'];dart_remaining[i]=dart_ticks
                            event_ids[i]=len(dart_events);perturbation_counts[i]+=1
                            dart_events.append(dict(id=cases[i]['id'],step=step,logical_expert_step=int(slot),
                                before_pose=np.r_[positions[i],rotations[i]].tolist(),**noise))
                for i,request in zip(selected,requests):
                    if dart_remaining[i]>0:
                        request['perturbation_pose']=dart_targets[i].tolist()
                        perturbation_mask[step,i]=True
            commands = client.query(requests)
            arm_term = env.action_manager.get_term('arm')
            measured_joints = env.scene['robot'].data.joint_pos[:, arm_term._joint_ids]
            joint_commands = measured_joints.clone() if joint_delta else None
            audit_start = len(audits)
            for i, request, command in zip(selected, requests, commands):
                last_expert_diagnostics[i] = command.get('diagnostic')
                if command['status'] != 'command_found':
                    results.append(dict(id=cases[i]['id'], initial_case=cases[i], reference=cases[i]['reference'],
                        split=splits[i], reason='expert_query_failed', step=step, expert_result=command))
                    active[i] = False
                    continue
                if command['diagnostic']['stage'] == 'approach_grasp':
                    stage[i] = True
                if joint_delta:
                    from garmi_rl.expert_actions import PARK
                    joint_commands[i] = torch.as_tensor(PARK+1.5*np.array(command['action'][3:]), device=env.device)
                    ys[step, i, :3] = command['action'][:3]
                else:
                    ys[step, i] = command['action']
                if step in (0, 30) and len(audits) < 32:
                    audits.append(dict(query=request, result=command, executed_action=ys[step, i].tolist()))
            if joint_delta:
                encoded = encode_joint_target(measured_joints, joint_commands)
                ys[step, :, 3:] = encoded.cpu().numpy()
                for audit in audits[audit_start:]:
                    audit['executed_action'] = ys[step, audit['query']['case']].tolist()
                    audit['joint_delta_conversion'] = COMMAND_VERSION
            teacher_stages[step] = stage
            with torch.no_grad():
                after, reward, done, truncated, _ = env.step(torch.tensor(ys[step], device=env.device))
            xs[step+1] = after['policy'].cpu().numpy()
            phases[step+1] = env.fixed_finish.phase.cpu().numpy()
            rewards[step] = reward.cpu().numpy()
            terminated[step] = done.cpu().numpy()
            truncateds[step] = truncated.cpu().numpy()
            bc[step] = approach_label_mask(phases[step:step+2], perturbation_mask[step:step+1])[0]
            valid = bc[step] & active
            if valid.any():
                error = float(np.max(np.abs(xs[step+1, valid, 39:39+action_dim]-ys[step, valid])))
                maximum_action_error = max(maximum_action_error, error)
                assert error < 1.e-6, ('Executed command not matched to label', error)
            assert np.isfinite(xs[step+1]).all() and np.isfinite(ys[step]).all()
            d = env.diagnostics
            ended = (done | truncated).cpu().numpy() & active
            if args.dart:
                normal_ticks += (active & (phase==0) & ~perturbation_mask[step])
                dart_remaining -= perturbation_mask[step].astype(int)
                completed_perturbation=perturbation_mask[step] & ((dart_remaining==0)|ended)
                if completed_perturbation.any():
                    hp,hq,_,_=mdp.geometry(env)
                    positions=(hp-env.scene.env_origins).cpu().numpy();rotations=hq.cpu().numpy()
                    for i in np.flatnonzero(completed_perturbation):
                        event=dart_events[event_ids[i]];before=np.asarray(event['before_pose'])
                        angle=(Rotation.from_quat(before[[4,5,6,3]]).inv()*Rotation.from_quat(rotations[i][[1,2,3,0]])).magnitude()
                        event.update(after_pose=np.r_[positions[i],rotations[i]].tolist(),
                            actual_translation_m=float(np.linalg.norm(positions[i]-before[:3])),
                            actual_rotation_rad=float(angle),terminated_during_perturbation=bool(ended[i]))
            ending_cases = env.grasp_bank.cases() if ended.any() else []
            for i in np.flatnonzero(ended):
                reason = next((key for key in ('collision','dropped','out_of_bounds','success','approach_timeout','attempt_failed') if bool(d[key][i])), 'timeout')
                case = cases[i]
                row = dict(id=case['id'], reference=case['reference'], initial_case=case, split=splits[i],
                    region=case['region'], depth=case['depth'], reason=reason, steps=step+1,
                    seconds=(step+1)*env.step_dt, command_version=COMMAND_VERSION,
                    handover_trigger=int(d['handover_trigger'][i]), handover_distance=float(d['handover_distance'][i]),
                    handover_angle=float(d['handover_angle'][i]),
                    close_start_distance=float(env.fixed_finish.close_start_distance[i]),
                    close_start_angle=float(env.fixed_finish.close_start_angle[i]),
                    final_phase=int(env.fixed_finish.phase[i]),
                    final_gripper_width=float(d['gripper_width'][i]),
                    height=float(d['height'][i]), contact=bool(d['contact'][i]), supported=bool(d['supported'][i]),
                    slip=float(d['slip'][i]), hold_seconds=float(env.grasp_state['hold'][i]),
                    teacher_reached_pregrasp=bool(stage[i]),
                    augmentation='ga_ddpg_dart' if args.dart else None,
                    perturbations=int(perturbation_counts[i]),
                    collision_bodies={body:float(f) for body,f in zip(BODIES,d['forbidden_force'][i]) if f>.1})
                if args.diagnostics and reason != 'success':
                    row['contact_targets'] = []
                    for body, key in zip(BODIES, env.forbidden_sensors):
                        sensor = env.scene[key]
                        forces = sensor.data.force_matrix_w_history[i].norm(dim=-1).amax(dim=(0, 1))
                        row['contact_targets'] += [
                            dict(body=body, target=target, force=float(force))
                            for target, force in zip(sensor.cfg.filter_prim_paths_expr, forces) if force > .1]
                    previous = next((q for q in requests if q['case'] == i), None)
                    if previous is not None:
                        failure_queries.append(dict(previous, id=case['id']+'-before-end'))
                    current = snapshot_queries(env, xs[step+1], cases, [int(i)], stage, 'failure', step+1)[0]
                    failure_queries.append(dict(current, id=case['id']+'-after-end'))
                row['final_snapshot'] = dict(**ending_cases[i],
                    observed_joint_positions=env.scene['robot'].data.joint_pos[i,env.observed_joint_ids].cpu().tolist(),
                    goal_relative_position=xs[step+1,i,30:33].tolist(),
                    goal_relative_quaternion=xs[step+1,i,33:37].tolist())
                row['last_expert_diagnostic'] = last_expert_diagnostics[i]
                row['strict_success_verified'] = strict_success(row)
                row['selected_for_dataset'] = bool(reason == 'success' and
                    (not args.require_strict_success or row['strict_success_verified']) and
                    accepted[splits[i]][case['reference']] < quotas[splits[i]] and
                    (not args.dart or perturbation_counts[i]>0))
                if reason == 'success':
                    assert row['height'] >= .08 and row['contact'] and not row['supported'] and row['slip'] < .02 and row['hold_seconds'] >= 2.
                    assert not np.any(np.diff(teacher_stages[:step+1,i].astype(int)) < 0)
                    path = out/'trajectories'/f"{case['id']}.npz"
                    np.savez_compressed(path, observations=xs[:step+2,i], actions=ys[:step+1,i],
                        rewards=rewards[:step+1,i], phases=phases[:step+2,i], bc_mask=bc[:step+1,i],
                        terminated=terminated[:step+1,i], truncated=truncateds[:step+1,i],
                        approach_stage=teacher_stages[:step+1,i],perturbation_mask=perturbation_mask[:step+1,i])
                    row.update(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bc_steps=int(bc[:step+1,i].sum()))
                    if row['selected_for_dataset']:
                        demonstrations[splits[i]].append(dict(row))
                        accepted[splits[i]][case['reference']] += 1
                results.append(row)
            active &= ~ended
            if step % 100 == 0:
                save('collecting', batch=batch, step=step, active=int(active.sum()))
                print('GISKARD_COLLECT', batch, step, int(active.sum()), dict(accepted['train']), flush=True)
            if not active.any():
                break
        assert not active.any(), 'Unfinished collection episodes'
        batch_timings.append(time.perf_counter()-batch_start)
        save('collecting', batches=len(batch_timings))
    assert complete(), f'Collection quota not reached: {accepted}'
    save('complete', batches=len(batch_timings))
    print('GISKARD_COLLECTION_DONE', dict(accepted['train']), dict(accepted['validation']), flush=True)
except BaseException as error:
    traceback.print_exc()
    save('failed', error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    client.close()
    env.close()
    app.close()
