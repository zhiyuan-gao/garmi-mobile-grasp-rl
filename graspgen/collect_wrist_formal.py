"""Supervise collection only: coverage, DART interface, ordinary and DART quotas."""
import argparse
import collections
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import yaml

from review_wrist_v6 import matching_condition, strict_approach

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/wrist-expert-collection-v6'
PILOT = ROOT/'runs/wrist-expert-pilot-v6'
STAGES = [
    dict(name='coverage', train=1, validation=1, dart=False, num_envs=48, max_batches=8),
    dict(name='dart-check', train=2, validation=1, dart=True, num_envs=100, max_batches=12),
    dict(name='expert-plain', train=250, validation=20, dart=False, num_envs=128, max_batches=500),
    dict(name='expert-dart', train=50, validation=5, dart=True, num_envs=128, max_batches=300),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def inside(path):
    return '/workspace/'+str(path.relative_to(ROOT))


def prepare():
    assert not OUT.exists(), 'Never overwrite the frozen collection plan'
    assert read(PILOT/'summary.json')['status'] == 'complete'
    OUT.mkdir()
    plan = dict(status='prepared', collection_authorized=True, training_authorized=False,
                target_train_per_reference=300, target_validation_per_reference=25,
                target_total_per_object=13000, target_total=26000,
                stages=STAGES, objects={}, code_sha256={}, source_sha256={},
                distribution='Same local XY +/-4 mm windows and successful PARK-arm starts as wrist-expert-pilot-v6',
                strict_thresholds=dict(handover_position_m=.005, handover_angle_rad=.05,
                    close_start_position_m=.005, close_start_angle_rad=.05,
                    minimum_lift_m=.08, maximum_slip_m=.02, minimum_hold_seconds=2.),
                gate_policy='Each object advances only after its coverage and DART gates pass. Exhausted stage budget stops that object and preserves partial successes.',
                split_policy='Reference and train/validation split assigned before execution; existing split assignments retained; held-out evaluation untouched')
    for name, seed in [('pitcher', 2026101981), ('mug', 2026101982)]:
        folder = OUT/name
        folder.mkdir()
        source = ROOT/'resources/validated-ycb-v6'/name
        evidence = read(source/'validation.json')
        for rel, digest in read(source/'SHA256.json').items():
            assert sha(source/rel) == digest, rel
        for rel, digest in evidence['runtime_code_sha256'].items():
            # Collector changes only affect sampling and acceptance; physics remains frozen.
            if rel != 'rl/collect_giskard.py':
                assert sha(ROOT/rel) == digest, rel
            plan['code_sha256'][rel] = sha(ROOT/rel)
        profile = yaml.safe_load((PILOT/name/'task.yaml').read_text())
        assert profile['garmi_task'] == yaml.safe_load((source/'task.yaml').read_text())['garmi_task']
        shutil.copy2(PILOT/name/'evaluation-cases.json', folder/'evaluation-cases.json')
        assert len(read(folder/'evaluation-cases.json')) == 80
        profile['evaluation'] = dict(cases=inside(folder/'evaluation-cases.json'),
                                    cases_sha256=sha(folder/'evaluation-cases.json'))
        (folder/'task.yaml').write_text(yaml.safe_dump(profile, sort_keys=False))
        lo, hi = np.array(profile['garmi_task']['kitchen']['base_bounds_map_xy'])
        spec = dict(object=name, xy_radius_m=.004, minimum_state_separation_m=.0002,
                    parallel_attempts_per_needed_success=4,
                    base_bounds_task_xy=[(np.array([lo[1]-6.35, .3-hi[0]])+.002).tolist(),
                                         (np.array([hi[1]-6.35, .3-lo[0]])-.002).tolist()],
                    references={}, excluded_case_files=[inside(source/'audit-cases.json'),
                        inside(PILOT/name/'collection-cases.json')],
                    excluded_result_files=[inside(PILOT/name/'collection/summary.json')])
        for row in evidence['references']:
            ref = row['ref']['id']
            proofs = [row['approach_evidence']] + row.get('additional_approach_evidence', [])
            windows = []
            for condition in row['conditions']:
                proof = next(p for p in proofs if matching_condition(condition, p['result']))
                assert strict_approach(proof['result'])
                case = dict(object=name, reference=ref,
                            object_state=condition['object_state'],
                            robot_state=proof['result']['initial_case']['robot_state'])
                windows.append(dict(id=condition['id'], case=case))
            spec['references'][ref] = windows
        assert len(spec['references']) == 40
        write(folder/'sampling-template.json', spec)
        for path in [source/'SHA256.json', source/'bank.json', source/'position-mask.json',
                     PILOT/name/'train-index.json', PILOT/name/'validation-index.json',
                     PILOT/name/'collection/summary.json', PILOT/name/'collection-cases.json',
                     source/'audit-cases.json', folder/'task.yaml', folder/'evaluation-cases.json',
                     folder/'sampling-template.json']:
            plan['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
        plan['objects'][name] = dict(seed=seed, configuration=inside(folder/'task.yaml'),
                                    initial_trajectories=read(PILOT/name/'report.json')['datasets'])
    for rel in ['graspgen/collect_wrist_formal.py', 'graspgen/review_wrist_v6.py',
                'rl/garmi_rl/expert_collection.py', 'rl/audit_giskard_data.py',
                'scripts/rl.sh', 'scripts/container_exec.sh']:
        plan['code_sha256'][rel] = sha(ROOT/rel)
    for rel in plan['code_sha256']:
        target = OUT/'code-snapshot'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, target)
    write(OUT/'code-snapshot/SHA256.json', plan['code_sha256'])
    write(OUT/'plan.json', plan)
    print('FORMAL_PLAN_FROZEN', plan['target_total'], flush=True)


def verify(plan):
    for field in ['code_sha256', 'source_sha256']:
        for rel, digest in plan[field].items():
            assert sha(ROOT/rel) == digest, rel
    assert plan['training_authorized'] is False


def run_logged(command, logfile, gpu):
    with logfile.open('w') as log:
        return subprocess.run(command, cwd=ROOT, env=dict(os.environ, GARMI_RL_GPU=gpu),
                              stdout=log, stderr=subprocess.STDOUT).returncode


def audit(folder, gpu, partial=False):
    prefix = 'partial-' if partial else ''
    for split in ['train', 'validation']:
        manifest = read(folder/f'{split}-index.json')
        for row in manifest['demonstrations']:
            assert strict_approach(row) and row['split'] == split
        if partial:
            manifest.update(status='complete', collection_quota_complete=False,
                            scope='Audit of available strict successes only; the collection gate did not pass')
            write(folder/f'{prefix}{split}-index.json', manifest)
    result = folder/f'{prefix}data-audit.json'
    command = ['bash', 'scripts/rl.sh', '/workspace/rl/audit_giskard_data.py',
               '--index', inside(folder/f'{prefix}train-index.json'),
               '--validation', inside(folder/f'{prefix}validation-index.json'), '--out', inside(result)]
    assert run_logged(command, folder/f'{prefix}data-audit.log', gpu) == 0
    assert read(result)['status'] == 'passed'
    return result


def run(name, gpu, resume=False, plan_path=None):
    folder = OUT/name
    lock = (folder/'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = plan_path or OUT/'plan.json'
    plan = read(plan_path)
    verify(plan)
    if resume:
        state = read(folder/'supervisor.json')
        assert state['status'] in ('stopped', 'failed', 'restarting')
        state.update(status='running', pid=os.getpid(), resumed_unix=time.time(), gpu=gpu)
        for field in ['error', 'failed_stage', 'returncode', 'collector_error', 'missing']:
            state.pop(field, None)
    else:
        assert not (folder/'supervisor.json').exists(), 'Existing collection requires explicit recovery, not a second supervisor'
        state = dict(status='running', object=name, gpu=gpu, pid=os.getpid(), started_unix=time.time(),
                     training_started=False, stages=[], target_train=12000, target_validation=1000)
    state.update(active_plan=str(plan_path.relative_to(ROOT)), active_plan_sha256=sha(plan_path))
    write(folder/'supervisor.json', state)
    refs = list(read(folder/'sampling-template.json')['references'])
    try:
        for number, stage in enumerate(plan['stages']):
            verify(plan)
            label = stage['name']
            target = folder/label
            if label in state['stages']:
                assert read(target/'summary.json')['status'] == 'complete'
                assert read(target/'data-audit.json')['status'] == 'passed'
                continue
            continuing = target.exists()
            assert not continuing or resume
            specification = folder/f'{label}-sampling.json'
            if continuing:
                previous = read(target/'summary.json')
                assert previous['status'] in ('collecting', 'failed')
                assert previous['local_sampling_sha256'] == sha(specification)
            else:
                spec = copy.deepcopy(read(folder/'sampling-template.json'))
                spec['excluded_result_files'] += [inside(folder/old/'summary.json') for old in state['stages']]
                write(specification, spec)
            seed = plan['objects'][name]['seed'] + 100*number + plan.get('seed_offset', 0)
            command = ['bash', 'scripts/rl.sh', '/workspace/rl/collect_giskard.py', '--headless',
                       '--config', inside(folder/'task.yaml'), '--out', inside(target),
                       '--num_envs', str(stage['num_envs']), '--workers', str(stage.get('workers', 8)),
                       '--success-per-reference', str(stage['train']),
                       '--validation-per-reference', str(stage['validation']),
                       '--max-batches', str(stage['max_batches']),
                       '--seed', str(seed),
                       '--local-sampling', inside(specification), '--require-strict-success']
            sources = []
            if label in ['coverage', 'expert-plain']:
                sources.append(PILOT/name)
            if label == 'expert-plain':
                sources.append(folder/'coverage')
            if label == 'expert-dart':
                sources.append(folder/'dart-check')
            if continuing:
                command += ['--resume', '--allow-parallel-resize']
            elif sources:
                for split in ['train', 'validation']:
                    command += [f'--bootstrap-{split}', *[inside(p/f'{split}-index.json') for p in sources]]
            if stage['dart']:
                command += ['--dart']
            state.update(current_stage=label, active_summary=str((target/'summary.json').relative_to(ROOT)))
            write(folder/'supervisor.json', state)
            suffix = f'-resume-{seed}' if continuing else ''
            write(folder/f'{label}{suffix}-launch.json', dict(command=command, sampling_sha256=sha(specification),
                  configuration_sha256=sha(folder/'task.yaml'), active_plan_sha256=sha(plan_path),
                  prior_summaries_sha256={str((folder/old/'summary.json').relative_to(ROOT)):
                                         sha(folder/old/'summary.json') for old in state['stages']}))
            rc = run_logged(command, folder/f'{label}{suffix}.log', gpu)
            if rc != 0:
                state.update(status='stopped', failed_stage=label, returncode=rc)
                if (target/'summary.json').exists():
                    summary = read(target/'summary.json')
                    state['collector_error'] = summary.get('error')
                    state['missing'] = {split: {ref: quota-summary['accepted'][split].get(ref, 0)
                        for ref in refs if summary['accepted'][split].get(ref, 0) < quota}
                        for split, quota in [('train', stage['train']), ('validation', stage['validation'])]}
                    state['partial_audit'] = str(audit(target, gpu, partial=True).relative_to(ROOT))
                return
            summary = read(target/'summary.json')
            assert summary['status'] == 'complete' and summary['require_strict_success']
            for split, quota in [('train', stage['train']), ('validation', stage['validation'])]:
                assert summary['accepted'][split] == {ref: quota for ref in refs}
            audit(target, gpu)
            state['stages'].append(label)
            write(folder/'supervisor.json', state)
        # Never reassign existing splits when merging the ordinary and DART stages.
        keys = set()
        for split, quota in [('train', 300), ('validation', 25)]:
            merged = read(folder/'expert-plain'/f'{split}-index.json')
            merged['demonstrations'] += read(folder/'expert-dart'/f'{split}-index.json')['demonstrations']
            counts = collections.Counter(row['reference'] for row in merged['demonstrations'])
            assert counts == {ref: quota for ref in refs}
            for row in merged['demonstrations']:
                key = tuple(row['initial_case']['robot_state'] + row['initial_case']['object_state'])
                assert key not in keys
                keys.add(key)
            merged['trajectories'] = len(merged['demonstrations'])
            merged['strict_handover_and_close_start_verified'] = True
            write(folder/f'{split}-index.json', merged)
        audit(folder, gpu)
        state.update(status='complete', train_trajectories=12000, validation_trajectories=1000)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['elapsed_seconds'] = time.time()-state['started_unix']
        write(folder/'supervisor.json', state)
        print('FORMAL_SUPERVISOR', name, state['status'], state.get('current_stage'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--object', choices=['pitcher', 'mug'])
    parser.add_argument('--gpu', choices=['0', '1'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        assert args.object and args.gpu
        run(args.object, args.gpu, args.resume, args.plan.resolve() if args.plan else None)
