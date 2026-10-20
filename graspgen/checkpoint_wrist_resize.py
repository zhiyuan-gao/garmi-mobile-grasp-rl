"""Checkpoint this project's collectors at batch boundaries, preserving all successes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/wrist-expert-collection-v6'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def command(pid):
    return (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')[:-1]


def collector_pid(name, stage):
    expected = f'/workspace/runs/wrist-expert-collection-v6/{name}/{stage}'.encode()
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            args = command(int(directory.name))
            if (args and args[0].endswith(b'python3') and b'/workspace/rl/collect_giskard.py' in args
                    and b'--out' in args and args[args.index(b'--out')+1] == expected):
                environment = (directory/'environ').read_bytes().split(b'\0')
                assert f'GARMI_RL_PROJECT_ROOT={ROOT}'.encode() in environment
                found.append(int(directory.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    assert len(found) == 1, found
    return found[0]


def main(name, revision):
    folder = OUT/name
    checkpoint = revision/name
    checkpoint.mkdir()
    state = read(folder/'supervisor.json')
    assert state['status'] == 'running' and state['current_stage'] == 'expert-plain'
    assert str(ROOT/'graspgen/collect_wrist_formal.py').encode() in command(state['pid'])
    pid = collector_pid(name, state['current_stage'])
    stage = folder/state['current_stage']
    baseline = None
    deadline = time.time()+1200
    write(checkpoint/'status.json', dict(status='waiting_for_two_batch_boundaries', collector_pid=pid,
                                       supervisor_pid=state['pid']))
    while time.time() < deadline:
        summary = read(stage/'summary.json')
        if summary['status'] != 'collecting':
            raise RuntimeError('Collector exited before checkpoint')
        if 'batches' not in summary:
            time.sleep(.08)
            continue
        if baseline is None:
            baseline = summary
            write(checkpoint/'baseline-start.json', baseline)
        elif summary['batches'] > baseline['batches']:
            # Freeze the main Python process first so the old supervisor cannot
            # race its shutdown. Only these exact, verified PIDs are signalled.
            os.kill(pid, signal.SIGSTOP)
            assert str(ROOT/'graspgen/collect_wrist_formal.py').encode() in command(state['pid'])
            os.kill(state['pid'], signal.SIGTERM)
            os.kill(pid, signal.SIGINT)
            os.kill(pid, signal.SIGCONT)
            write(checkpoint/'baseline-end.json', summary)
            break
        time.sleep(.08)
    else:
        raise TimeoutError('No complete batch boundary within 20 minutes; collection left running')
    for _ in range(180):
        try:
            args = command(pid)
        except (FileNotFoundError, ProcessLookupError):
            break
        if not args:
            break
        time.sleep(1)
    else:
        raise TimeoutError('Collector did not finish its graceful checkpoint')
    final = read(stage/'summary.json')
    assert final['status'] == 'failed' and 'KeyboardInterrupt' in final['error']
    before = {r['id'] for r in summary['results']}
    assert before <= {r['id'] for r in final['results']}
    saved = {}
    for split in ['train', 'validation']:
        index = read(stage/f'{split}-index.json')
        rows = index['demonstrations']
        assert len(rows) == sum(final['accepted'][split].values())
        saved[split] = len(rows)
        for row in rows:
            path = ROOT/row['file'].removeprefix('/workspace/')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        shutil.copy2(stage/f'{split}-index.json', checkpoint/f'{split}-index.json')
    shutil.copy2(stage/'summary.json', checkpoint/'summary.json')
    write(checkpoint/'old-supervisor.json', state)
    duration = summary['wall_seconds']-baseline['wall_seconds']
    attempts = len(summary['results'])-len(baseline['results'])
    successes = sum(map(sum, [list(v.values()) for v in summary['accepted'].values()])) - sum(
        map(sum, [list(v.values()) for v in baseline['accepted'].values()]))
    write(checkpoint/'status.json', dict(status='checkpointed', collector_pid=pid, saved=saved,
        baseline_seconds=duration, baseline_attempts=attempts, baseline_selected=successes,
        baseline_attempts_per_hour=attempts/duration*3600,
        baseline_selected_per_hour=successes/duration*3600,
        checkpoint_reason='Explicit parallel resize at a complete batch boundary',
        completed_trajectory_hashes_verified=True))
    state.update(status='restarting', restart_reason='parallel_resize', checkpoint=str(checkpoint.relative_to(ROOT)))
    write(folder/'supervisor.json', state)
    print('CHECKPOINTED', name, saved, attempts, 'attempts in', duration, 'seconds', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('object', choices=['pitcher','mug'])
    parser.add_argument('--revision', required=True, type=Path)
    args = parser.parse_args()
    main(args.object, args.revision.resolve())
