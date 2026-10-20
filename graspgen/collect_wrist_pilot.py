"""Freeze and collect a finite 200-attempt pilot per v6 object; never train."""
import argparse
import collections
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import yaml

from review_wrist_v6 import matching_condition, strict_approach

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/wrist-expert-pilot-v6'
N_ENVS = 100


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inside(path):
    return '/workspace/' + str(path.relative_to(ROOT))


def state(case):
    return np.array(case['robot_state'] + case['object_state'])


def prepare():
    assert not OUT.exists(), 'Frozen pilot already exists; do not replace its splits'
    OUT.mkdir()
    plan = dict(status='frozen', attempts_per_reference=5, attempts_per_object=200,
                training_attempts_per_object=150, validation_attempts_per_object=50,
                heldout_evaluation_cases_per_object=80, training_authorized=False,
                collection_authorized=True, num_envs=N_ENVS, workers=8,
                initial_arm='Standard PARK; no endpoint arm-state restoration',
                distribution='Object XY +/-4 mm about verified window centers; base XY +/-4 mm about successful approach starts, clipped inside legal bounds',
                split_scope='New initial states with the same reference library and local placement windows; not held-out objects, grasps, or regions',
                selection='Splits fixed before execution; publish strict physical successes only; retain every failed attempt in reports',
                objects={})
    for name, seed in [('pitcher', 2026101971), ('mug', 2026101972)]:
        folder = OUT / name
        folder.mkdir()
        source = ROOT / 'resources/validated-ycb-v6' / name
        for path, digest in read(source/'SHA256.json').items():
            assert sha(source/path) == digest, path
        evidence = read(source/'validation.json')
        for path, digest in evidence['runtime_code_sha256'].items():
            assert sha(ROOT/path) == digest, path
        profile = yaml.safe_load((source/'task.yaml').read_text())
        refs = profile['garmi_task']['multi_object']['reference_ids'][name]
        assert len(refs) == len(set(refs)) == 40
        lo, hi = np.array(profile['garmi_task']['kitchen']['base_bounds_map_xy'])
        base_lo = np.array([lo[1]-6.35, .3-hi[0]]) + .002
        base_hi = np.array([hi[1]-6.35, .3-lo[0]]) - .002
        used = [state(c) for c in read(source/'audit-cases.json')]
        rngs = {s: np.random.default_rng(seed + offset) for s, offset in
                [('train', 0), ('validation', 100), ('evaluation', 200)]}
        cases, frozen, assignments = [], [], []
        for number, reference in enumerate(evidence['references']):
            ref = reference['ref']['id']
            assert ref == refs[number]
            proofs = [reference['approach_evidence']] + reference.get('additional_approach_evidence', [])
            for trial in range(7):
                index = len(cases)
                split = ('evaluation' if trial >= 5 else
                         'validation' if index % 4 == 0 else 'train')
                condition = reference['conditions'][trial % len(reference['conditions'])]
                proof = next(p for p in proofs if matching_condition(condition, p['result']))
                assert strict_approach(proof['result'])
                for _ in range(1000):
                    case = dict(id=f'{name}-{split}-{number:02d}-{trial:02d}',
                                object=name, reference=ref,
                                robot_state=copy.deepcopy(proof['result']['initial_case']['robot_state']),
                                object_state=copy.deepcopy(condition['object_state']))
                    case['robot_state'][:2] = np.clip(
                        np.array(case['robot_state'][:2]) + rngs[split].uniform(-.004, .004, 2),
                        base_lo, base_hi).tolist()
                    case['object_state'][:2] = (np.array(case['object_state'][:2]) +
                                               rngs[split].uniform(-.004, .004, 2)).tolist()
                    # More than float32 scene-coordinate rounding from all prior states.
                    if all(np.max(np.abs(state(case)-old)) > .00025 for old in used):
                        break
                else:
                    raise RuntimeError('Could not sample a separated case')
                assert matching_condition(condition, {'initial_case': case})
                assert np.allclose(case['robot_state'][3:7], [1, 0, 0, 0])
                assert not np.any(case['robot_state'][7:]) and not np.any(case['object_state'][7:])
                used.append(state(case))
                if split == 'evaluation':
                    frozen.append(case)
                else:
                    cases.append(case)
                    assignments.append(dict(index=index, id=case['id'], reference=ref, split=split,
                                            condition=condition['id']))
        assert len(cases) == 200 and len(frozen) == 80
        assert set(collections.Counter(c['reference'] for c in cases).values()) == {5}
        assert collections.Counter(a['split'] for a in assignments) == {'train': 150, 'validation': 50}
        for ref in refs:
            assert {a['split'] for a in assignments if a['reference'] == ref} == {'train', 'validation'}
        write(folder/'collection-cases.json', cases)
        write(folder/'split-assignments.json', assignments)
        write(folder/'evaluation-cases.json', frozen)
        for split in ['train', 'validation']:
            write(folder/f'{split}-cases.json', [c for c, a in zip(cases, assignments) if a['split'] == split])
        profile['evaluation'] = dict(cases=inside(folder/'evaluation-cases.json'),
                                    cases_sha256=sha(folder/'evaluation-cases.json'))
        (folder/'task.yaml').write_text(yaml.safe_dump(profile, sort_keys=False))
        inputs = {str(p.relative_to(ROOT)): sha(p) for p in folder.iterdir() if p.is_file()}
        inputs.update(evidence['runtime_code_sha256'])
        inputs[str(source.relative_to(ROOT)/'SHA256.json')] = sha(source/'SHA256.json')
        inputs['graspgen/collect_wrist_pilot.py'] = sha(Path(__file__))
        inputs['rl/audit_giskard_data.py'] = sha(ROOT/'rl/audit_giskard_data.py')
        write(folder/'inputs-sha256.json', inputs)
        plan['objects'][name] = dict(seed=seed, configuration_sha256=sha(folder/'task.yaml'),
                                   frozen_inputs_sha256=sha(folder/'inputs-sha256.json'))
    write(OUT/'plan.json', plan)
    print('FROZEN', {name: dict(attempts=200, train=150, validation=50, heldout=80)
                     for name in plan['objects']}, flush=True)


def verify_inputs(name):
    folder = OUT/name
    plan = read(OUT/'plan.json')
    assert plan['training_authorized'] is False and plan['collection_authorized'] is True
    assert sha(folder/'inputs-sha256.json') == plan['objects'][name]['frozen_inputs_sha256']
    for path, digest in read(folder/'inputs-sha256.json').items():
        assert sha(ROOT/path) == digest, path
    return folder, plan


def collect(name, gpu):
    folder, plan = verify_inputs(name)
    raw = folder/'collection'
    assert not raw.exists(), 'Refuse to overwrite or silently retry a finite pilot'
    command = ['bash', 'scripts/rl.sh', '/workspace/rl/collect_giskard.py', '--headless',
               '--config', inside(folder/'task.yaml'), '--out', inside(raw),
               '--num_envs', str(N_ENVS), '--workers', str(plan['workers']),
               '--audit-cases', inside(folder/'collection-cases.json'),
               '--seed', str(plan['objects'][name]['seed'])]
    status = dict(status='collecting', object=name, command=command, gpu=gpu,
                  started_unix=time.time(), training_started=False)
    write(folder/'status.json', status)
    try:
        with (folder/'collection.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, GARMI_RL_GPU=gpu),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        report(name)
        # The data auditor imports torch; the CPU planner venv does not contain it.
        command = ['bash', 'scripts/rl.sh',
                   '/workspace/rl/audit_giskard_data.py', '--index', inside(folder/'train-index.json'),
                   '--validation', inside(folder/'validation-index.json'), '--out', inside(folder/'data-audit.json')]
        with (folder/'data-audit.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, GARMI_RL_GPU=gpu),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        audit = read(folder/'data-audit.json')
        assert audit['status'] == 'passed'
        status.update(status='complete', result=read(folder/'report.json'), data_audit=audit)
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status['elapsed_seconds'] = time.time()-status['started_unix']
        write(folder/'status.json', status)
    print('PILOT_COMPLETE', name, status['result']['counts'], flush=True)


def report(name):
    folder, plan = verify_inputs(name)
    raw = read(folder/'collection/summary.json')
    assert raw['status'] == 'complete' and len(raw['results']) == 200
    assert raw['configuration_sha256'] == plan['objects'][name]['configuration_sha256']
    cases = read(folder/'collection-cases.json')
    assignments = read(folder/'split-assignments.json')
    heldout = read(folder/'evaluation-cases.json')
    selected = {'train': [], 'validation': []}
    counts = collections.Counter()
    refs = collections.defaultdict(collections.Counter)
    attempts, seen = [], set()
    for row in raw['results']:
        match = re.search(r'-b(\d+)-c(\d+)$', row['id'])
        index = int(match[1])*N_ENVS + int(match[2])
        assert index not in seen and 0 <= index < 200
        seen.add(index)
        expected, assignment = cases[index], assignments[index]
        assert row['reference'] == expected['reference'] and row['split'] == assignment['split']
        delta = float(np.max(np.abs(state(row['initial_case'])-state(expected))))
        assert delta < 5.e-5, (index, delta)
        assert all(np.max(np.abs(state(row['initial_case'])-state(c))) > .0002 for c in heldout)
        passed = strict_approach(row)
        reason = 'strict_success' if passed else ('non_strict_success' if row['reason'] == 'success' else row['reason'])
        counts[reason] += 1
        refs[row['reference']][reason] += 1
        if passed:
            assert row['selected_for_dataset']
            path = ROOT/row['file'].removeprefix('/workspace/')
            assert sha(path) == row['sha256']
            chosen = copy.deepcopy(row)
            chosen['frozen_case_id'] = expected['id']
            chosen['strict_success_verified'] = True
            selected[row['split']].append(chosen)
        attempts.append(dict(frozen_case_id=expected['id'], collector_id=row['id'],
                             reference=row['reference'], split=row['split'], reason=reason,
                             dataset_selected=passed, reset_error_m=delta))
    assert len(seen) == 200
    for split, demonstrations in selected.items():
        write(folder/f'{split}-index.json', dict(status='complete', expert_backend='giskard',
              command_version=raw['command_version'], demonstrations=demonstrations,
              trajectories=len(demonstrations), physical_success_verified=True,
              strict_handover_and_close_start_verified=True,
              selection='All strict successes from a finite pilot, split frozen before execution'))
    write(folder/'attempts.json', attempts)
    rows = []
    for ref in [c['reference'] for c in cases[::5]]:
        c = refs[ref]
        assert sum(c.values()) == 5
        rows.append(dict(reference=ref, attempts=5, strict_success=c['strict_success'],
                         train=sum(r['reference'] == ref for r in selected['train']),
                         validation=sum(r['reference'] == ref for r in selected['validation']),
                         failures={k: v for k, v in c.items() if k != 'strict_success'}))
    report = dict(status='complete', object=name, attempts=200, counts=dict(counts),
                  success_rate=counts['strict_success']/200,
                  datasets={s: len(v) for s, v in selected.items()},
                  successful_references=sum(r['strict_success'] > 0 for r in rows),
                  zero_success_references=[r['reference'] for r in rows if not r['strict_success']],
                  per_reference=rows, heldout_evaluation_cases=80,
                  heldout_evaluation_executed=False, training_started=False,
                  configuration_sha256=raw['configuration_sha256'],
                  collection_summary_sha256=sha(folder/'collection/summary.json'),
                  scope=plan['distribution'], split_scope=plan['split_scope'])
    write(folder/'report.json', report)
    with (folder/'per-reference.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('PILOT_REPORT', name, dict(counts), report['datasets'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['prepare', 'collect', 'report'])
    parser.add_argument('--object', choices=['mug', 'pitcher'])
    parser.add_argument('--gpu', choices=['0', '1'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'collect':
        assert args.object and args.gpu
        collect(args.object, args.gpu)
    else:
        assert args.object
        report(args.object)
