"""Collection-only bowl workflow for published YCB banks; never runs training.

Run with the host graspgen venv after the nominal pilot. Every stage is retained,
and a failed coverage gate stops before quota collection. Completed stages can
be reused; interrupted quota collection requires the collector's explicit resume.
"""
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def inside(path):
    return '/workspace/' + str(path.resolve().relative_to(ROOT))


def host(path):
    return ROOT / path.removeprefix('/workspace/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', required=True, choices=['mug', 'pitcher'])
    parser.add_argument('--root', type=Path, default=ROOT/'runs/ycb-expert-collection-v1')
    parser.add_argument('--gpu', required=True, choices=['0', '1'])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.root.resolve()/args.object
    config = out/'task.yaml'
    profile = yaml.safe_load(config.read_text())
    refs = profile['garmi_task']['multi_object']['reference_ids'][args.object]
    plan = read(args.root/'plan.json')
    assert plan['training_authorized'] is False
    assert hashlib.sha256(config.read_bytes()).hexdigest() == plan['objects'][args.object]['configuration_sha256']
    assert len(refs) == len(set(refs)) == 40
    initial = read(host(profile['expert_preflight']['audit_cases']))
    nominal = collections.defaultdict(list)
    for case in initial:
        if case['id'].endswith('-0'):
            nominal[case['reference']].append(case)
    frozen = read(host(profile['evaluation']['cases']))
    key = lambda c: tuple(c['robot_state']+c['object_state'])
    used = {key(c) for c in frozen}
    rng = random.Random(2026101852)
    cases = []
    for ref in refs:
        for index in range(10):
            case = copy.deepcopy(nominal[ref][index % len(nominal[ref])])
            case['id'] = f'full-approach-{ref}-{index:02d}'
            # Interior of the validated +/-5 mm object windows. The robot
            # starts parked, as in the ordinary collector, never at an IK endpoint.
            for axis in (0, 1):
                case['object_state'][axis] += rng.uniform(-.0045, .0045)
            assert key(case) not in used
            used.add(key(case))
            cases.append(case)
    case_path = out/'expert-check-cases-400.json'
    if case_path.exists():
        assert read(case_path) == cases
    else:
        write(case_path, cases)
    if args.prepare_only:
        print('PREPARED', args.object, len(cases), 'full-approach cases')
        return
    env = dict(os.environ, GARMI_RL_GPU=args.gpu)
    state = dict(status='running', object=args.object, started=time.time(),
                 optimizer_updates=0, stages=[], configuration_sha256=plan['objects'][args.object]['configuration_sha256'])

    def run(name, script, argv, output):
        state['current_stage'] = name
        write(out/'collection-status.json', state)
        if output.exists():
            assert read(output)['status'] in ('complete', 'passed'), f'Incomplete stage needs explicit recovery: {output}'
        else:
            with (out/f'{name}.log').open('w') as log:
                subprocess.run(['bash', str(ROOT/'scripts/rl.sh'), '/workspace/rl/'+script, *map(str, argv)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['stages'].append(name)
        write(out/'collection-status.json', state)
        return read(output)

    def collect(name, train, validation, seed, dart=False):
        argv = ['--headless', '--config', inside(config), '--out', inside(out/name),
                '--num_envs', '128', '--workers', '8', '--balance-collection',
                '--success-per-reference', train, '--validation-per-reference', validation,
                '--max-batches', '1000', '--seed', seed, '--focus-references', *refs]
        if dart:
            argv += ['--dart']
        report = run(name, 'collect_giskard.py', argv, out/name/'summary.json')
        assert report['status'] == 'complete'
        for split, quota in [('train', train), ('validation', validation)]:
            counts = collections.Counter(r['reference'] for r in read(out/name/f'{split}-index.json')['demonstrations'])
            assert set(counts) == set(refs) and set(counts.values()) == {quota}

    try:
        pilot = read(out/'pilot/summary.json')
        assert pilot['status'] == 'complete', 'Nominal pilot must finish first'
        audit = run('expert-check-400', 'collect_giskard.py', [
            '--headless', '--config', inside(config), '--out', inside(out/'expert-check-400'),
            '--num_envs', '128', '--workers', '8', '--audit-cases', inside(case_path),
            '--seed', '2026101853'], out/'expert-check-400/summary.json')
        run('preflight-data-audit', 'audit_giskard_data.py', [
            '--index', inside(out/'expert-check-400/train-index.json'),
            '--validation', inside(out/'expert-check-400/validation-index.json'),
            '--out', inside(out/'preflight-data-audit.json')], out/'preflight-data-audit.json')
        successful = {r['reference'] for r in audit['results'] if r['reason'] == 'success'}
        missing = sorted(set(refs)-successful)
        write(out/'coverage-gate.json', dict(status='failed' if missing else 'passed',
              successful_references=len(successful), missing_references=missing,
              results=audit['counts'], required='At least one complete physical success per reference in 400 cases'))
        assert not missing, f'Full approach gate failed for {len(missing)}/40 references'
        collect('interface-collection', 2, 1, 2026101854, dart=True)
        run('interface-data-audit', 'audit_giskard_data.py', [
            '--index', inside(out/'interface-collection/train-index.json'),
            '--validation', inside(out/'interface-collection/validation-index.json'),
            '--out', inside(out/'interface-data-audit.json')], out/'interface-data-audit.json')
        collect('expert-plain', 250, 20, 2026101855)
        collect('expert-dart', 50, 5, 2026101856, dart=True)
        split_keys = {}
        for split, quota in [('train', 300), ('validation', 25)]:
            merged = copy.deepcopy(read(out/'expert-plain'/f'{split}-index.json'))
            merged['demonstrations'] += read(out/'expert-dart'/f'{split}-index.json')['demonstrations']
            merged['trajectories'] = len(merged['demonstrations'])
            counts = collections.Counter(r['reference'] for r in merged['demonstrations'])
            assert set(counts) == set(refs) and set(counts.values()) == {quota}
            split_keys[split] = {key(r['initial_case']) for r in merged['demonstrations']}
            assert len(split_keys[split]) == merged['trajectories']
            assert not split_keys[split] & {key(c) for c in frozen}
            write(out/f'base-{split}-index.json', merged)
        assert not split_keys['train'] & split_keys['validation']
        run('full-data-audit', 'audit_giskard_data.py', [
            '--index', inside(out/'base-train-index.json'), '--validation', inside(out/'base-validation-index.json'),
            '--out', inside(out/'full-data-audit.json')], out/'full-data-audit.json')
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='stopped', error=str(error))
        raise
    finally:
        state['elapsed_seconds'] = time.time()-state['started']
        write(out/'collection-status.json', state)


if __name__ == '__main__':
    main()
