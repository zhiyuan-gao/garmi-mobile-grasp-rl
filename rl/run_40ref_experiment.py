"""Run the fixed 40-reference bowl expansion, then 8000 new PPO updates.

Host supervisor: GPU stages are serial, Giskard uses eight CPU workers.
Demonstration validation and frozen policy evaluations never become labels.
"""
import argparse
import collections
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'bootstrap'
OLD_CASES = OLD / 'regression-cases.json'
CRAM = '/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python'
CONTAINER = ['bash', str(ROOT / 'scripts/container_exec.sh')]


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def inside(path):
    return '/workspace/' + str(Path(path).resolve().relative_to(ROOT))


def host(path):
    return ROOT / str(path).removeprefix('/workspace/') if str(path).startswith('/workspace/') else Path(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_key(case):
    return tuple(case['robot_state'] + case['object_state'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume a stopped queue, preserving completed stages and collected data')
    args = parser.parse_args()
    out = args.out.resolve()
    prep = read(out / 'preparation.json')
    refs = prep['references']
    task = out / 'task.yaml'
    config = yaml.safe_load(task.read_text())
    assert len(refs) == 40 and len(set(refs)) == 40
    assert refs == config['garmi_task']['multi_object']['reference_ids']['bowl']
    assert prep['ppo_updates'] == 8000 and prep['ppo_num_envs'] == 4096
    for relative, expected in read(ROOT / 'assets/manifest.json')['files_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    for key, expected in config['garmi_task']['kitchen']['files_sha256'].items():
        assert sha(host(config['garmi_task']['kitchen'][key])) == expected, key
    if prep['checkpoint']:
        assert sha(host(prep['checkpoint'])) == prep['checkpoint_sha256']
    assert len(read(out / 'evaluation-cases-4000.json')) == 4000
    assert sha(out / 'evaluation-cases-4000.json') == config['evaluation']['cases_sha256']
    if prep['bootstrap']:
        assert len(read(OLD_CASES)) == 1200
    if args.check_only:
        print('PLAN_CHECK_PASSED', len(refs), 'references, 8000 updates, 4096 environments')
        return

    lock = (out / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        state = read(out / 'supervisor.json')
        assert state['status'] in ('failed', 'interrupted')
        recovery_id = state.get('resume_count', 0) + 1
        archive = out / f'recovery-{recovery_id:02d}'
        archive.mkdir()
        for name in ('supervisor.json', 'REPORT.md', 'queue.log', state['current_stage'] + '.log'):
            if (out/name).exists(): shutil.copy2(out/name, archive/name)
        state.setdefault('interruptions', []).append({k:state.get(k) for k in
            ('status','current_stage','error','traceback','wall_seconds')})
        state.pop('error', None)
        state.pop('traceback', None)
        state.update(status='restarting', host_pid=os.getpid(), restarted=time.time(), resume_count=recovery_id)
    else:
        assert not (out / 'supervisor.json').exists(), 'Use --resume for an explicitly stopped queue'
        state = dict(status='starting', host_pid=os.getpid(), started=time.time(), stages=[], evaluations=[],
                     planned_updates=8000, num_envs=4096, reference_count=40, resume_count=0)

    def save():
        state['wall_seconds'] = time.time() - state['started']
        state['active_stage_seconds'] = sum(s.get('seconds', time.time()-s['started'] if s['status']=='running' else 0.)
                                            for s in state['stages'])
        atomic(out / 'supervisor.json', state)
        report = '# 碗：40 个固定参考，专家示范＋纠偏＋PPO\n\n'
        report += f"状态：`{state['status']}`；当前阶段：`{state.get('current_stage', 'starting')}`。\n\n"
        if state.get('resume_count'):
            report += f"发生过中断，已恢复 {state['resume_count']} 次。原中断信息保存在 recovery-*/；总经过时间包含停机，实际阶段运行约 {state['active_stage_seconds']/3600:.2f} 小时。\n\n"
        report += f"初始化：{'导入的四参考模型' if prep['bootstrap'] else '新模型'}；同一个策略学习全部 40 个参考。\n\n"
        report += '正式示范目标每参考 300 条；三轮策略状态纠偏后，再做 8000 次 PPO 更新。预检结果见 expert-check-400/summary.json。\n\n'
        report += 'PPO 使用 4096 个环境，每 500 次保存并串行评估 4000 个固定场景。8000 是本轮新增更新数，容量检查的 8 次包含在内。\n\n'
        report += '按旧速度，纯 PPO 约 13 小时；数据准备、模仿学习和评估另计，不设置 10 小时强制停止。\n\n'
        for e in state['evaluations']:
            report += f"- {e['name']}：{e['counts']}，成功率 {e['success_rate']:.1%}。\n"
        if state.get('best_main'):
            best = state['best_main']
            report += f"\n目前主评估最好：{best['name']}，{best['success_rate']:.1%}。\n"
        if state.get('error'):
            report += '\n停止原因：`' + state['error'] + '`。\n'
        report += '\n[完整阶段状态](supervisor.json) · [专家预检](expert-check-400/summary.json)\n'
        (out / 'REPORT.md').write_text(report)

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Supervisor received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def cleanup(child):
        # Docker exec clients do not reliably forward signals. First allow the
        # existing PPO supervisor to checkpoint, then target only this run's tree.
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
        code = """import os, signal
from pathlib import Path
needle = %r
project = %r
own = os.getpid()
processes = {}
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) == own: continue
    try:
        argv = (p/'cmdline').read_bytes().split(b'\\0')
        # Apptainer shares host PIDs: another project can also use /workspace.
        # Only seed cleanup with processes carrying this checkout's marker.
        environment = (p/'environ').read_bytes().split(b'\\0')
        stat = (p/'stat').read_text().split()
        processes[int(p.name)] = (int(stat[3]), argv, environment)
    except (OSError, ValueError, IndexError): pass
targets = {pid for pid, (_, argv, environment) in processes.items()
           if ('GARMI_RL_PROJECT_ROOT=' + project).encode() in environment
           and any(a.decode(errors='replace').startswith(needle) for a in argv)}
while True:
    expanded = targets | {pid for pid, (parent, _, _) in processes.items() if parent in targets}
    if expanded == targets: break
    targets = expanded
for pid in targets:
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print('Scoped cleanup', sorted(targets))
""" % (inside(out), str(ROOT))
        subprocess.run(CONTAINER + ['/usr/bin/python3', '-c', code],
                       timeout=30, check=False, stdout=subprocess.DEVNULL)

    def run(name, script, argv, marker, cpu=False, host_script=False, timeout=14400):
        if args.resume and any(s['name'] == name and s['status'] == 'complete' for s in state['stages']):
            print('STAGE_REUSED', name, flush=True)
            return
        if args.resume and (script != 'collect_giskard.py' or '--resume' not in argv):
            partial = out / name
            if partial.is_dir():
                archive = out / f'recovery-{state["resume_count"]:02d}' / name
                shutil.move(str(partial), str(archive))
        if host_script:
            command = [sys.executable, '-u', str(ROOT / 'rl' / script)]
        elif cpu:
            command = CONTAINER + [CRAM, '/workspace/rl/' + script]
        else:
            command = [str(ROOT / 'scripts/rl.sh'), '/workspace/rl/' + script]
        command += list(map(str, argv))
        row = dict(name=name, command=command, started=time.time(), status='running', timeout_seconds=timeout)
        state['stages'].append(row)
        state.update(status='running', current_stage=name)
        save()
        log_path = out / (name + '.log')
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            save()
            try:
                while child.poll() is None:
                    if time.time() - row['started'] > timeout:
                        raise TimeoutError(f'{name} exceeded its explicit timeout')
                    time.sleep(5)
                    save()
            except BaseException:
                cleanup(child)
                raise
        output = log_path.read_text(errors='replace')
        bad = [word for word in ('CUDA error:', 'out of memory', 'buffer overflow', 'PhysX error')
               if word.lower() in output.lower()]
        passed = child.returncode == 0 and not bad and (not marker or marker in output)
        row.update(status='complete' if passed else 'failed', returncode=child.returncode,
                   seconds=time.time() - row['started'], errors=bad)
        save()
        if not passed:
            raise RuntimeError(f'{name} failed; inspect {log_path.name}')
        print('STAGE_DONE', name, round(row['seconds'], 1), flush=True)

    def evaluate(name, checkpoint, cases):
        run(name, 'evaluate_kitchen.py', ['--headless', '--config', inside(task), '--checkpoint', checkpoint,
            '--cases', inside(cases), '--num_envs', '240', '--seed', '12345', '--out', inside(out / name)],
            'EVALUATION_DONE', timeout=2400)
        report = read(out / name / 'summary.json')
        assert report['status'] == 'complete' and report['completed'] == len(read(cases))
        assert report['evaluation_cases_sha256'] == sha(cases)
        assert report['checkpoint_sha256'] == sha(host(checkpoint))
        entry = dict(name=name, checkpoint=checkpoint, cases=str(cases), **{
            k: report[k] for k in ('success_rate', 'counts', 'per_reference')})
        state['evaluations'].append(entry)
        if cases.name == 'evaluation-cases-4000.json':
            update_best(entry)
        save()

    def update_best(entry):
        if entry['success_rate'] > state.get('best_main', {}).get('success_rate', -1):
            state['best_main'] = entry
            atomic(out / 'best-main.json', entry)

    def bc(name, train_index, validation_index, checkpoint, epochs=100):
        run(name, 'train_expert_bc.py', ['--config', inside(task), '--index', inside(train_index),
            '--validation', inside(validation_index), *(['--init-checkpoint', checkpoint] if checkpoint else []),
            '--epochs', epochs, '--patience', '1000', '--batch-size', '1024', '--learning-rate', '.0003',
            '--loss', 'action_mse', '--balance-references', '--seed', '2026101820', '--out', inside(out / name)], 'BC_DONE')
        assert read(out / name / 'summary.json')['status'] == 'complete'
        return inside(out / name / 'last.pt')

    def collect(name, train_quota, val_quota, seed, dart=False):
        recovering = (out/name/'summary.json').exists() and read(out/name/'summary.json')['status'] != 'complete'
        if recovering:
            assert args.resume
            seed += 100000 * state['resume_count']
        argv = ['--headless', '--config', inside(task), '--out', inside(out / name), '--num_envs', '128',
                '--workers', '8', '--balance-collection', '--focus-references', *prep['new_references'],
                '--success-per-reference', train_quota, '--validation-per-reference', val_quota,
                '--max-batches', '1000', '--seed', seed]
        if dart:
            argv += ['--dart']
        if recovering:
            argv += ['--resume']
        run(name, 'collect_giskard.py', argv, 'GISKARD_COLLECTION_DONE', timeout=21600)
        report = read(out / name / 'summary.json')
        assert report['status'] == 'complete'
        for split, quota in [('train', train_quota), ('validation', val_quota)]:
            rows = read(out / name / f'{split}-index.json')['demonstrations']
            counts = collections.Counter(r['reference'] for r in rows)
            assert set(counts) == set(prep['new_references']) and set(counts.values()) == {quota}, counts

    def merge_data(split, total_per_reference):
        data = (copy.deepcopy(read(OLD / f'base-{split}-index.json')) if prep['bootstrap'] else
                dict(read(out / 'expert-plain' / f'{split}-index.json'), demonstrations=[]))
        for name in ('expert-plain', 'expert-dart'):
            data['demonstrations'] += read(out / name / f'{split}-index.json')['demonstrations']
        data['trajectories'] = len(data['demonstrations'])
        counts = collections.Counter(r['reference'] for r in data['demonstrations'])
        assert set(counts) == set(refs) and set(counts.values()) == {total_per_reference}, counts
        data.update(status='complete', source_indices=([str(OLD / f'base-{split}-index.json')] if prep['bootstrap'] else []) + [
            str(out / 'expert-plain' / f'{split}-index.json'), str(out / 'expert-dart' / f'{split}-index.json')])
        atomic(out / f'base-{split}-index.json', data)
        return data

    def correction_cases(train, val):
        pools = {}
        for split, data in [('train', train), ('validation', val)]:
            pool = collections.defaultdict(list)
            for row in data['demonstrations']:
                pool[row['reference']].append(row['initial_case'])
            pools[split] = pool
        rng = random.Random(2026101821)
        heldout = {case_key(c) for c in (read(OLD_CASES) if prep['bootstrap'] else []) + read(out / 'evaluation-cases-4000.json')}
        train_keys = {case_key(r['initial_case']) for r in train['demonstrations']}
        val_keys = {case_key(r['initial_case']) for r in val['demonstrations']}
        assert not train_keys & val_keys and not heldout & (train_keys | val_keys)
        validation_cases = [c for ref in refs for c in rng.sample(pools['validation'][ref], 10)]
        rng.shuffle(validation_cases)
        for round_id in range(1, 4):
            training_cases = [c for ref in refs for c in rng.sample(pools['train'][ref], 50)]
            rng.shuffle(training_cases)
            for split, cases in [('train', training_cases), ('validation', validation_cases)]:
                for chunk in range(0, len(cases), 200):
                    atomic(out / f'round-{round_id:02d}-{split}-{chunk//200:02d}-cases.json', cases[chunk:chunk+200])
        atomic(out / 'split-audit.json', dict(status='passed', train_trajectories=len(train_keys),
            validation_trajectories=len(val_keys), test_cases=len(heldout), train_validation_overlap=0,
            data_test_overlap=0, correction_train_cases_per_round=2000, correction_validation_cases_per_round=400))

    def correction(name, cases, checkpoint, seed):
        run(name, 'evaluate_kitchen.py', ['--headless', '--config', inside(task), '--checkpoint', checkpoint,
            '--cases', inside(cases), '--num_envs', '200', '--out', inside(out / name), '--seed', seed,
            '--dagger-queries', '--dagger-stochastic', '--giskard-stages', '--dagger-probability', '.5',
            '--dagger-min-step', '5', '--dagger-max-step', '600'], 'EVALUATION_DONE', timeout=2400)
        query_path = out / name / 'dagger-queries.json'
        assert len(read(query_path)) > 0
        run(name + '-expert', 'giskard_expert_service.py', ['--requests', inside(query_path),
            '--out', inside(out / (name + '-expert.json')), '--workers', '8'], 'GISKARD_BATCH_DONE', cpu=True, timeout=3600)
        run(name + '-labels', 'build_giskard_labels.py', ['--queries', inside(query_path),
            '--expert-results', inside(out / (name + '-expert.json')), '--out', inside(out / (name + '-labels'))],
            'GISKARD_LABELS_DONE', timeout=1200)
        labels = read(out / (name + '-labels/index.json'))
        assert labels['status'] == 'passed' and labels['labels'] > 0
        return labels['dagger_label_sets']

    try:
        run('reset-check', 'check_collection_reset.py', ['--headless', '--config', inside(task),
            '--out', inside(out / 'reset-check'), '--num_envs', '128', '--resets', '300'],
            'COLLECTION_RESET_CHECK_DONE')
        run('expert-check-400', 'collect_giskard.py', ['--headless', '--config', inside(task),
            '--out', inside(out / 'expert-check-400'), '--num_envs', '128', '--workers', '8',
            '--audit-cases', inside(out / 'expert-check-cases-400.json')], 'GISKARD_COLLECTION_DONE')
        audit = read(out / 'expert-check-400/summary.json')
        assert audit['status'] == 'complete'
        successful_refs = {r['reference'] for r in audit['results'] if r['reason'] == 'success'}
        assert successful_refs == set(refs), ('References with no successful precheck', set(refs)-successful_refs)
        run('interface-collection', 'collect_giskard.py', ['--headless', '--config', inside(task),
            '--out', inside(out / 'interface-collection'), '--num_envs', '128', '--workers', '8',
            '--balance-collection', '--dart', '--success-per-reference', '2', '--validation-per-reference', '1',
            '--max-batches', '100', '--seed', '2026101809'], 'GISKARD_COLLECTION_DONE')
        run('interface-data-audit', 'audit_giskard_data.py', ['--index', inside(out / 'interface-collection/train-index.json'),
            '--validation', inside(out / 'interface-collection/validation-index.json'),
            '--out', inside(out / 'interface-data-audit.json')], 'GISKARD_DATA_AUDIT')
        assert read(out / 'interface-data-audit.json')['status'] == 'passed'
        smoke_checkpoint = bc('interface-bc', out / 'interface-collection/train-index.json',
            out / 'interface-collection/validation-index.json', None, epochs=2)
        smoke_cfg = copy.deepcopy(config)
        smoke_cfg['expert_bc'] = dict(index=inside(out / 'interface-collection/train-index.json'), coefficient=3.,
            batch_size=1024, state_normalization='frozen', sampling='uniform_references', data_device='cpu')
        (out / 'interface-ppo.yaml').write_text(yaml.safe_dump(smoke_cfg))
        run('interface-ppo', 'train_trial.py', ['--headless', '--config', inside(out / 'interface-ppo.yaml'),
            '--init-checkpoint', smoke_checkpoint, '--num_envs', '64', '--updates', '2',
            '--skip-evaluation', '--out', inside(out / 'interface-ppo')], 'TRIAL_DONE', timeout=1200)
        assert read(out / 'interface-ppo/progress.json')['status'] == 'complete'
        atomic(out / 'interface-tests.json', dict(status='passed'))
        source = out / (f"source-recovery-{state['resume_count']:02d}" if args.resume else 'source')
        source.mkdir()
        paths = list((ROOT / 'rl/garmi_rl').glob('*.py')) + [ROOT / 'rl' / n for n in (
            'collect_giskard.py', 'train_expert_bc.py', 'evaluate_kitchen.py', 'giskard_expert_service.py',
            'build_giskard_labels.py', 'audit_giskard_data.py', 'train_trial.py', 'run_trial.py',
            'evaluate_trial_checkpoint.py', 'prepare_40ref_trial.py', 'prepare_40ref_initialization.py',
            'run_40ref_experiment.py', 'check_collection_reset.py')]
        hashes = {}
        for path in paths:
            relative = path.relative_to(ROOT / 'rl')
            dest = source / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            hashes[str(relative)] = sha(path)
        atomic(out / (source.name + '-sha256.json'), hashes)
        state.setdefault('source_versions', []).append(dict(directory=str(source), sha256=str(out/(source.name+'-sha256.json'))))
        # Balanced expert quotas are independent of the test set or policy score.
        collect('expert-plain', 250, 20, 2026101810)
        collect('expert-dart', 50, 5, 2026101811, dart=True)
        original = merge_data('train', 300)
        validation = merge_data('validation', 25)
        run('full-data-audit', 'audit_giskard_data.py', ['--index', inside(out / 'base-train-index.json'),
            '--validation', inside(out / 'base-validation-index.json'), '--out', inside(out / 'full-data-audit.json')],
            'GISKARD_DATA_AUDIT')
        assert read(out / 'full-data-audit.json')['status'] == 'passed'
        correction_cases(original, validation)
        init_checkpoint = None
        if prep['checkpoint']:
            run('normalization', 'prepare_40ref_initialization.py', ['--index', inside(out / 'base-train-index.json'),
                '--checkpoint', prep['checkpoint'], '--config', inside(task), '--out', inside(out / 'normalization')],
                'NORMALIZATION_40REF_DONE')
            assert read(out / 'normalization/summary.json')['status'] == 'passed'
            init_checkpoint = inside(out / 'normalization/actor-init.pt')
            evaluate('initial-model-main4000', prep['checkpoint'], out / 'evaluation-cases-4000.json')
        grouped = collections.defaultdict(list)
        for case in read(out / 'evaluation-cases-4000.json'):
            grouped[case['reference']].append(case)
        quick = [case for ref in refs for case in grouped[ref][:10]]
        atomic(out / 'evaluation-cases-400.json', quick)
        checkpoint = bc('initial-bc', out / 'base-train-index.json', out / 'base-validation-index.json',
                        init_checkpoint)
        evaluate('initial-bc-quick400', checkpoint, out / 'evaluation-cases-400.json')
        train_labels = read(OLD / 'round-03-train-index.json')['dagger_label_sets'] if prep['bootstrap'] else []
        val_labels = read(OLD / 'round-03-validation-index.json')['dagger_label_sets'] if prep['bootstrap'] else []
        for round_id in range(1, 4):
            prefix = f'round-{round_id:02d}'
            for split, chunks, labels in [('train', 10, train_labels), ('validation', 2, val_labels)]:
                for chunk in range(chunks):
                    name = f'{prefix}-{split}-{chunk:02d}'
                    labels += correction(name, out / (name + '-cases.json'), checkpoint,
                        2026101830 + 100 * round_id + 20 * (split == 'validation') + chunk)
            training = dict(original, dagger_label_sets=train_labels, dagger_labels=sum(x['count'] for x in train_labels))
            validating = dict(validation, dagger_label_sets=val_labels, dagger_labels=sum(x['count'] for x in val_labels))
            # All new and reused corrective labels must retain their source split.
            train_ids = {r['id'] for r in original['demonstrations']}
            val_ids = {r['id'] for r in validation['demonstrations']}
            for sets, ids in [(train_labels, train_ids), (val_labels, val_ids)]:
                for labels in sets:
                    assert all(s['initial_case'].get('source_initial_id', s['initial_case']['id']) in ids
                               for s in labels['samples'])
            atomic(out / (prefix + '-train-index.json'), training)
            atomic(out / (prefix + '-validation-index.json'), validating)
            state['dagger_labels'] = training['dagger_labels']
            checkpoint = bc(prefix + '-bc', out / (prefix + '-train-index.json'),
                out / (prefix + '-validation-index.json'), checkpoint)
            if round_id < 3:
                evaluate(prefix + '-quick400', checkpoint, out / 'evaluation-cases-400.json')
        evaluate('pre-ppo-main4000', checkpoint, out / 'evaluation-cases-4000.json')
        if prep['bootstrap']:
            evaluate('pre-ppo-original1200', checkpoint, OLD_CASES)
        ppo_config = copy.deepcopy(config)
        ppo_config['expert_bc'] = dict(coefficient=3., batch_size=1024, state_normalization='frozen')
        ppo_config['expert_bc'].update(index=inside(out / 'round-03-train-index.json'),
            data_device='cpu', sampling='uniform_references')
        (out / 'ppo.yaml').write_text(yaml.safe_dump(ppo_config))
        assert ppo_config['expert_bc']['coefficient'] == 3
        # Count the capacity smoke's 8 updates in the 8000-update budget. Full
        # optimizer, RNG, scaler and scheduler state is resumed at every boundary.
        for target in [8] + list(range(500, 8001, 500)):
            name = f'ppo-{target:04d}'
            argv = ['--out', str(out / name), '--config', str(out / 'ppo.yaml'), '--title', f'40 references PPO {target}',
                '--num_envs', '4096', '--updates', target, '--eval-updates', target, '--eval-envs', '240',
                '--defer-evaluation', '--skip-videos']
            argv += ['--init-checkpoint' if target == 8 else '--resume', checkpoint]
            if target == 8000 and prep['bootstrap']:
                argv += ['--paired-eval-cases', inside(OLD_CASES), '--paired-eval-envs', '240', '--paired-eval-seed', '12345']
            run(name, 'run_trial.py', argv, None, host_script=True, timeout=10800)
            supervision = read(out / name / 'supervisor.json')
            progress = read(out / name / 'progress.json')
            assert supervision['status'] == progress['status'] == 'complete' and not supervision['errors']
            assert progress['update'] == target
            checkpoint = progress['latest_checkpoint']
            main_eval = [e for e in progress['evaluations'] if e['update'] == target][-1]
            entry = dict(name=name, **main_eval)
            state['evaluations'].append(entry)
            state.update(ppo_update=target, latest_checkpoint=checkpoint)
            update_best(entry)
            save()
        state.update(status='complete', latest_checkpoint=checkpoint)
        save()
        print('FORTY_REFERENCE_EXPERIMENT_DONE', flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=repr(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
