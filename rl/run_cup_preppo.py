"""Cup: physical expert audit, demonstrations, BC and three DAgger rounds.

This queue deliberately has no PPO training stage. It uses the same action,
reward, physics, demonstration quotas and BC settings as the bowl experiment.
"""
import argparse
import collections
import copy
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
CRAM = '/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python'
SCRIPTS = {
    'check_collection_reset.py', 'collect_giskard.py', 'audit_giskard_data.py',
    'train_expert_bc.py', 'evaluate_kitchen.py', 'giskard_expert_service.py',
    'build_giskard_labels.py',
}


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path):
    return '/workspace/' + str(Path(path).resolve().relative_to(ROOT))


def host(path):
    return ROOT / str(path).removeprefix('/workspace/')


def case_key(case):
    return tuple(case['robot_state'] + case['object_state'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'runs/cup4-left-preppo')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    out = args.out.resolve()
    task = out/'task.yaml'
    prep = read(out/'preparation.json')
    config = yaml.safe_load(task.read_text())
    refs = prep['references']
    assert prep['stop_before_ppo'] and prep['ppo_updates'] == 0
    assert len(refs) == len(set(refs)) == 4
    assert config['garmi_task']['multi_object']['objects'] == ['cup']
    assert refs == config['garmi_task']['multi_object']['reference_ids']['cup']
    assert sha(task) == prep['task_sha256']
    assert len(read(out/'evaluation-cases.json')) == 400
    assert sha(out/'evaluation-cases.json') == config['evaluation']['cases_sha256']
    for manifest in (ROOT/'assets/manifest.json', ROOT/'assets/cup/expert.json'):
        for relative, expected in read(manifest)['files_sha256'].items():
            assert sha(ROOT/relative) == expected, relative
    for key, expected in config['garmi_task']['kitchen']['files_sha256'].items():
        assert sha(host(config['garmi_task']['kitchen'][key])) == expected
    if args.check_only:
        print('CUP_PREPPO_CHECK_PASSED', refs, 'PPO updates: 0')
        return

    lock = (out/'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        state = read(out/'supervisor.json')
        assert state['status'] in ('failed', 'interrupted')
        state.setdefault('interruptions', []).append({
            key: state.get(key) for key in ('status', 'current_stage', 'error', 'traceback')})
        state.pop('error', None)
        state.pop('traceback', None)
        state['resume_count'] = state.get('resume_count', 0) + 1
    else:
        assert not (out/'supervisor.json').exists(), 'Use --resume after inspecting a stopped queue'
        state = dict(started=time.time(), stages=[], evaluations=[], resume_count=0)
    state.update(status='starting', host_pid=os.getpid(), ppo_updates=0,
                 stop_before_ppo=True, references=refs)
    container = os.environ.get('GARMI_RL_CONTAINER', 'garmi-cup-rl')

    def save():
        state['updated'] = time.time()
        state['wall_seconds'] = state['updated'] - state['started']
        atomic(out/'supervisor.json', state)
        report = '# 杯子：4 个抓法，验证到 PPO 之前\n\n'
        report += f"状态：{state['status']}；当前阶段：{state.get('current_stage', 'starting')}。\n\n"
        report += 'PPO 更新数：**0**。此队列不包含 PPO 训练或 PPO 短测。\n\n'
        report += '流程：80 次专家实抓预检 → 每抓法 250 条普通和 50 条扰动成功示范'
        report += '（另留 25 条验证）→ 初始 BC → 3 轮策略状态专家标注和 BC → 独立评估、录像。\n\n'
        report += '初始 BC 与每轮纠偏 BC 各训练 100 个 epoch；每阶段用同一组 400 个独立场景评估，评估状态不用于专家标注。\n\n'
        for e in state['evaluations']:
            report += f"- {e['name']}：{e['counts']}，成功率 {e['success_rate']:.1%}。\n"
        if state.get('error'):
            report += f"\n停止原因：{state['error']}。\n"
        (out/'REPORT.md').write_text(report)

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def cleanup(child):
        # This container belongs only to this cup experiment. Stop its workload
        # on interruption rather than leave a detached docker-exec job running.
        subprocess.run(['docker', 'stop', '--time', '10', container],
                       timeout=30, check=False, stdout=subprocess.DEVNULL)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()

    def run(name, script, argv, marker, cpu=False, timeout=14400):
        assert script in SCRIPTS, 'Only audited pre-PPO programs are permitted'
        if any(s['name'] == name and s['status'] == 'complete' for s in state['stages']):
            return
        log_path = out/(name+'.log')
        if args.resume:
            archive = out/f"recovery-{state['resume_count']:02d}"
            archive.mkdir(exist_ok=True)
            if (out/name).exists():
                shutil.move(str(out/name), str(archive/name))
            if log_path.exists():
                shutil.copy2(log_path, archive/log_path.name)
        command = ['docker', 'exec', '-e', 'PYTHONPATH=/workspace/rl',
                   '-e', 'GARMI_EXPERT_CONFIG=/workspace/assets/cup/expert.json',
                   '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1',
                   container, CRAM if cpu else '/isaac-sim/python.sh',
                   '/workspace/rl/'+script, *map(str, argv)]
        row = dict(name=name, command=command, status='running', started=time.time(),
                   timeout_seconds=timeout)
        state['stages'].append(row)
        state.update(status='running', current_stage=name)
        save()
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            row['pid'] = child.pid
            save()
            try:
                while child.poll() is None:
                    if time.time()-row['started'] > timeout:
                        raise TimeoutError(f'{name} exceeded {timeout} seconds')
                    time.sleep(5)
                    save()
            except BaseException:
                cleanup(child)
                raise
        output = log_path.read_text(errors='replace')
        errors = [w for w in ('CUDA error:', 'out of memory', 'buffer overflow', 'PhysX error')
                  if w.lower() in output.lower()]
        passed = child.returncode == 0 and marker in output and not errors
        row.update(status='complete' if passed else 'failed', returncode=child.returncode,
                   seconds=time.time()-row['started'], errors=errors)
        save()
        if not passed:
            raise RuntimeError(f'{name} failed; inspect {log_path.name}')
        print('STAGE_DONE', name, round(row['seconds'], 1), flush=True)

    def bc(name, train, validation, checkpoint=None, epochs=100):
        run(name, 'train_expert_bc.py', [
            '--config', inside(task), '--index', inside(train), '--validation', inside(validation),
            *(['--init-checkpoint', checkpoint] if checkpoint else []),
            '--epochs', epochs, '--patience', 1000, '--batch-size', 1024,
            '--learning-rate', .0003, '--loss', 'action_mse', '--balance-references',
            '--seed', 2026102004, '--out', inside(out/name)], 'BC_DONE')
        assert read(out/name/'summary.json')['status'] == 'complete'
        return inside(out/name/'last.pt')

    def evaluate(name, checkpoint):
        run(name, 'evaluate_kitchen.py', ['--headless', '--config', inside(task),
            '--checkpoint', checkpoint, '--cases', inside(out/'evaluation-cases.json'),
            '--num_envs', 200, '--seed', 2026102005, '--out', inside(out/name)], 'EVALUATION_DONE')
        result = read(out/name/'summary.json')
        assert result['status'] == 'complete' and result['completed'] == 400
        assert result['checkpoint_sha256'] == sha(host(checkpoint))
        assert result['evaluation_cases_sha256'] == sha(out/'evaluation-cases.json')
        entry = dict(name=name, checkpoint=checkpoint, **{
            k: result[k] for k in ('counts', 'success_rate', 'per_reference')})
        state['evaluations'] = [e for e in state['evaluations'] if e['name'] != name] + [entry]
        save()

    def collect(name, train, val, seed, dart=False):
        run(name, 'collect_giskard.py', ['--headless', '--config', inside(task),
            '--out', inside(out/name), '--num_envs', 128, '--workers', 8,
            '--balance-collection', '--success-per-reference', train,
            '--validation-per-reference', val, '--max-batches', 100, '--seed', seed,
            *(['--dart'] if dart else [])], 'GISKARD_COLLECTION_DONE', timeout=21600)
        result = read(out/name/'summary.json')
        assert result['status'] == 'complete'
        for split, quota in [('train', train), ('validation', val)]:
            rows = read(out/name/f'{split}-index.json')['demonstrations']
            counts = collections.Counter(r['reference'] for r in rows)
            assert set(counts) == set(refs) and set(counts.values()) == {quota}

    def data_audit(name, train, validation):
        run(name, 'audit_giskard_data.py', ['--index', inside(train),
            '--validation', inside(validation), '--out', inside(out/(name+'.json'))],
            'GISKARD_DATA_AUDIT')
        assert read(out/(name+'.json'))['status'] == 'passed'

    def correction(name, checkpoint, seed):
        cases = out/(name+'-cases.json')
        run(name, 'evaluate_kitchen.py', ['--headless', '--config', inside(task),
            '--checkpoint', checkpoint, '--cases', inside(cases), '--num_envs', 200,
            '--out', inside(out/name), '--seed', seed, '--dagger-queries',
            '--dagger-stochastic', '--giskard-stages', '--dagger-probability', .5,
            '--dagger-min-step', 5, '--dagger-max-step', 600], 'EVALUATION_DONE')
        queries = out/name/'dagger-queries.json'
        assert read(queries)
        run(name+'-expert', 'giskard_expert_service.py', ['--requests', inside(queries),
            '--out', inside(out/(name+'-expert.json')), '--workers', 8],
            'GISKARD_BATCH_DONE', cpu=True)
        run(name+'-labels', 'build_giskard_labels.py', ['--queries', inside(queries),
            '--expert-results', inside(out/(name+'-expert.json')),
            '--out', inside(out/(name+'-labels'))], 'GISKARD_LABELS_DONE')
        result = read(out/(name+'-labels')/'index.json')
        assert result['status'] == 'passed' and result['labels'] > 0
        assert set(result['per_reference']) == set(refs), 'Missing corrective labels for a reference'
        return result['dagger_label_sets']

    try:
        # Save the exact sources before any physics or learning, not after it.
        source = out/f"source-{state['resume_count']:02d}"
        source.mkdir(exist_ok=False)
        paths = list((ROOT/'rl/garmi_rl').glob('*.py')) + [
            ROOT/'rl'/name for name in sorted(SCRIPTS | {'run_cup_preppo.py'})]
        hashes = {}
        for path in paths:
            relative = path.relative_to(ROOT)
            dest = source/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            hashes[str(relative)] = sha(path)
        atomic(source/'sha256.json', hashes)
        state['source'] = str(source)
        # Accept the manually observed preflight only after its exact completion marker.
        reset = out/'reset-check/summary.json'
        if reset.exists() and not any(s['name'] == 'reset-check' for s in state['stages']):
            assert read(reset)['status'] == 'passed'
            assert 'COLLECTION_RESET_CHECK_DONE' in (out/'reset-check.log').read_text()
            state['stages'].append(dict(name='reset-check', status='complete', externally_checked=True))
        run('reset-check', 'check_collection_reset.py', ['--headless', '--config', inside(task),
            '--out', inside(out/'reset-check'), '--num_envs', 128, '--resets', 300],
            'COLLECTION_RESET_CHECK_DONE')
        run('expert-audit', 'collect_giskard.py', ['--headless', '--config', inside(task),
            '--out', inside(out/'expert-audit'), '--num_envs', 80, '--workers', 8,
            '--audit-cases', inside(out/'expert-audit-cases.json')], 'GISKARD_COLLECTION_DONE')
        result = read(out/'expert-audit/summary.json')
        assert result['status'] == 'complete' and len(result['results']) == 80
        audit = {ref: dict(collections.Counter(r['reason'] for r in result['results']
                                             if r['reference'] == ref)) for ref in refs}
        atomic(out/'expert-audit-per-reference.json', audit)
        # Stop for inspection if any reference cannot pass even half of the
        # sampled normal initial states; this is a data-quality gate, not reward.
        assert all(counts.get('success', 0) >= 10 for counts in audit.values()), (
            'Expert audit below 10/20 for a reference; inspect before collecting', audit)
        collect('interface-collection', 2, 1, 2026102006, dart=True)
        data_audit('interface-data-audit', out/'interface-collection/train-index.json',
                   out/'interface-collection/validation-index.json')
        bc('interface-bc', out/'interface-collection/train-index.json',
           out/'interface-collection/validation-index.json', epochs=2)
        collect('expert-plain', 250, 20, 2026102007)
        collect('expert-dart', 50, 5, 2026102008, dart=True)
        data = {}
        for split, quota in [('train', 300), ('validation', 25)]:
            plain = read(out/'expert-plain'/f'{split}-index.json')
            merged = copy.deepcopy(plain)
            merged['demonstrations'] += read(out/'expert-dart'/f'{split}-index.json')['demonstrations']
            merged['trajectories'] = len(merged['demonstrations'])
            counts = collections.Counter(r['reference'] for r in merged['demonstrations'])
            assert set(counts) == set(refs) and set(counts.values()) == {quota}
            atomic(out/f'base-{split}-index.json', merged)
            data[split] = merged
        data_audit('full-data-audit', out/'base-train-index.json', out/'base-validation-index.json')
        keys = {split: {case_key(r['initial_case']) for r in dataset['demonstrations']}
                for split, dataset in data.items()}
        test = {case_key(c) for c in read(out/'evaluation-cases.json')}
        assert not keys['train'] & keys['validation']
        assert not (keys['train'] | keys['validation']) & test
        rng = random.Random(2026102009)
        pools = {split: {ref: [r['initial_case'] for r in dataset['demonstrations']
                              if r['reference'] == ref] for ref in refs}
                 for split, dataset in data.items()}
        for r in range(1, 4):
            for split, count in [('train', 50), ('validation', 10)]:
                cases = [c for ref in refs for c in rng.sample(pools[split][ref], count)]
                rng.shuffle(cases)
                atomic(out/f'round-{r:02d}-{split}-cases.json', cases)
        atomic(out/'split-audit.json', dict(status='passed', train_trajectories=len(keys['train']),
            validation_trajectories=len(keys['validation']), test_cases=len(test),
            train_validation_overlap=0, data_test_overlap=0))
        checkpoint = bc('initial-bc', out/'base-train-index.json', out/'base-validation-index.json')
        evaluate('initial-bc-evaluation', checkpoint)
        labels = {'train': [], 'validation': []}
        for r in range(1, 4):
            prefix = f'round-{r:02d}'
            for split in labels:
                labels[split] += correction(prefix+'-'+split, checkpoint,
                    2026102020+10*r+(split == 'validation'))
                ids = {row['id'] for row in data[split]['demonstrations']}
                assert all(s['initial_case'].get('source_initial_id', s['initial_case']['id']) in ids
                           for entry in labels[split] for s in entry['samples'])
                combined = dict(data[split], dagger_label_sets=labels[split],
                                dagger_labels=sum(entry['count'] for entry in labels[split]))
                atomic(out/f'{prefix}-{split}-index.json', combined)
            checkpoint = bc(prefix+'-bc', out/f'{prefix}-train-index.json',
                            out/f'{prefix}-validation-index.json', checkpoint)
            evaluate(prefix+'-evaluation', checkpoint)
        state['final_checkpoint'] = checkpoint
        final = read(out/'round-03-evaluation/summary.json')
        # Record a small, explicit selection; don't substitute successful replay
        # results for the independent 400-case evaluation.
        selected = []
        for success in (True, False):
            choices = [row for row in final['results'] if (row['reason'] == 'success') == success]
            selected += choices[:2]
        videos = []
        for i, row in enumerate(selected):
            name = f'video-{i+1:02d}'
            run(name, 'evaluate_kitchen.py', ['--headless', '--enable_cameras',
                '--config', inside(task), '--checkpoint', checkpoint,
                '--cases', inside(out/'evaluation-cases.json'),
                '--case-index', row['case'], '--num_envs', 1,
                '--camera-eye', 2.6, -1.4, 2.1, '--camera-lookat', 1.1, 0., 1.05,
                '--out', inside(out/name)], 'EVALUATION_DONE')
            videos.append(dict(directory=name, evaluation_result=row,
                               replay=read(out/name/'summary.json')['counts']))
        atomic(out/'videos.json', videos)
        cards = []
        for item in videos:
            for video in (out/item['directory']).glob('*.mp4'):
                label = html.escape(f"{item['evaluation_result']['reference']}: {item['replay']}")
                cards.append(f'<article><p>{label}</p><video controls src="{video.relative_to(out)}"></video></article>')
        (out/'review.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>杯子抓取：PPO 前验证</title>'
            '<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto}'
            'video{width:100%;max-width:560px}article{display:inline-block;margin:12px}</style>'
            '<h1>杯子：4 个抓法，PPO 前验证</h1>'
            f"<p>最终 400 次独立评估成功率 {final['success_rate']:.1%}；PPO 更新数为 0。</p>"
            + ''.join(cards))
        state.update(status='complete', current_stage='stopped_before_ppo',
                     finished=time.time(), ppo_updates=0)
        save()
        print('CUP_PREPPO_DONE', final['counts'], flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
