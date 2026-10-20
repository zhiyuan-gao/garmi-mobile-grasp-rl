"""Drain the current mug segment, audit/evaluate update 2000, then stop.

Freeze only the original queue supervisor. Its independently launched training
process keeps running to its existing 2000-update limit. No frozen code changes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import train_wrist_ppo as base

ROOT = base.ROOT
FOLDER = base.OUT / 'mug'
OUT = FOLDER / 'stop-after-2000'


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=pid, state=fields[0], start_ticks=fields[19],
                    command=(path / 'cmdline').read_bytes().decode().split('\0')[:-1])
    except FileNotFoundError:
        return None


def alive(identity):
    current = process(identity['pid'])
    return bool(current and current['start_ticks'] == identity['start_ticks']
                and current['state'] not in ('Z', 'X'))


def inspect():
    state = base.read(FOLDER / 'supervisor.json')
    assert state['status'] == 'running' and state['current_stage'] == 'ppo-2000'
    row = state['stages'][-1]
    assert row['name'] == 'ppo-2000' and row['status'] == 'running'
    cmd = row['command']
    assert cmd[cmd.index('--updates') + 1] == '2000'
    progress = base.read(FOLDER / 'ppo-2000/progress.json')
    assert progress['status'] == 'training' and progress['update'] < 2000
    parent, launcher, trainer = map(process, [state['pid'], row['pid'], progress['pid']])
    assert parent and launcher and trainer
    assert any(x.endswith('/train_wrist_ppo.py') for x in parent['command'])
    assert parent['command'][parent['command'].index('--object') + 1] == 'mug'
    assert any(x.endswith('/train_trial.py') for x in trainer['command'])
    assert trainer['command'][trainer['command'].index('--out') + 1] == base.inside(FOLDER / 'ppo-2000')
    assert os.getpgid(parent['pid']) != os.getpgid(trainer['pid'])
    return state, dict(parent=parent, launcher=launcher, trainer=trainer,
                      current_update=progress['update'], stop_update=2000,
                      evaluate_after_training=True)


def run():
    state, request = inspect()
    plan = base.read(ROOT / state['active_plan'])
    base.verify(plan)
    OUT.mkdir(exist_ok=False)
    lock = (OUT / 'guardian.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.write(OUT / 'original-supervisor.json', state)
    request.update(requested_unix=time.time(), guardian_pid=os.getpid(),
                   status='arming', script_sha256=base.sha(Path(__file__)))
    base.write(OUT / 'request.json', request)
    parent_fd = os.pidfd_open(request['parent']['pid'])
    assert process(request['parent']['pid'])['start_ticks'] == request['parent']['start_ticks']
    signal.pidfd_send_signal(parent_fd, signal.SIGSTOP)
    for _ in range(100):
        if process(request['parent']['pid'])['state'] == 'T':
            break
        time.sleep(.02)
    assert process(request['parent']['pid'])['state'] == 'T'
    # Read once more after freezing so a final supervisor heartbeat is preserved.
    state = base.read(FOLDER / 'supervisor.json')
    assert state['current_stage'] == 'ppo-2000'
    state.update(status='stopping_after_2000', stop_after_update=2000,
                 boundary_guardian_pid=os.getpid(), original_supervisor_suspended=True)
    base.write(FOLDER / 'supervisor.json', state)
    request.update(status='waiting_for_update_2000', supervisor_frozen=True,
                   training_process_continues=alive(request['trainer']))
    base.write(OUT / 'request.json', request)
    print('STOP_AFTER_2000_ARMED', json.dumps(request), flush=True)

    def stage(label, script, args, marker):
        base.verify(plan)
        assert not any(x['name'] == label for x in state['stages'])
        command = ['bash', 'scripts/rl.sh', '/workspace/rl/' + script] + list(map(str, args))
        row = dict(name=label, status='running', command=command, started_unix=time.time(),
                   controller='stop_mug_after_2000')
        state['stages'].append(row)
        state.update(status='finalizing_stop_after_2000', current_stage=label)
        base.write(FOLDER / 'supervisor.json', state)
        with (FOLDER / (label + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, GARMI_RL_GPU='1'),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row['pid'] = child.pid
            base.write(FOLDER / 'supervisor.json', state)
            code = child.wait()
        text = (FOLDER / (label + '.log')).read_text(errors='replace')
        bad = [s for s in ('Traceback (most recent call last)', 'CUDA error:', 'out of memory',
                          'buffer overflow', 'PhysX error') if s.lower() in text.lower()]
        row.update(returncode=code, errors=bad, seconds=time.time()-row['started_unix'],
                   status='complete' if code == 0 and marker in text and not bad else 'failed')
        destination = base.host(command[command.index('--out')+1])
        result = destination if destination.is_file() else destination / 'summary.json'
        if result.exists():
            row['output_sha256'] = {str(result.relative_to(ROOT)): base.sha(result)}
        base.write(FOLDER / 'supervisor.json', state)
        assert row['status'] == 'complete', (label, code, bad)

    try:
        while alive(request['trainer']) or alive(request['launcher']):
            progress = base.read(FOLDER / 'ppo-2000/progress.json')
            assert progress['update'] <= 2000
            request.update(last_update=progress['update'], updated_unix=time.time())
            base.write(OUT / 'request.json', request)
            time.sleep(15)
        summary = base.read(FOLDER / 'ppo-2000/summary.json')
        assert summary['status'] == 'complete' and summary['update'] == 2000
        assert summary['frozen_state_normalization_verified']
        checkpoint = base.host(summary['latest_checkpoint'])
        latest = base.read(FOLDER / 'ppo-2000/latest.json')
        assert latest['update'] == 2000 and base.sha(checkpoint) == latest['sha256']
        assert checkpoint.with_suffix('.resume.pt').exists()
        text = (FOLDER / 'ppo-2000.log').read_text(errors='replace')
        assert 'TRIAL_DONE' in text
        assert not any(s.lower() in text.lower() for s in
                       ('Traceback (most recent call last)', 'CUDA error:', 'out of memory', 'buffer overflow', 'PhysX error'))
        row = state['stages'][-1]
        row.update(status='complete', seconds=time.time()-row['started_unix'],
                   completion_evidence='complete summary, TRIAL_DONE, checkpoint hash; adopted process exit code unavailable',
                   resource_measurements_end_at_stop_request=True,
                   output_sha256={str(p.relative_to(ROOT)): base.sha(p) for p in
                       [FOLDER/'ppo-2000/summary.json', FOLDER/'ppo-2000/latest.json',
                        checkpoint, checkpoint.with_suffix('.resume.pt')]})
        state.update(ppo_updates=2000, checkpoint=base.inside(checkpoint))
        # SIGTERM has the default disposition in the original supervisor. Resume
        # only to deliver termination, after the independently running job exited.
        for signum in (signal.SIGTERM, signal.SIGCONT):
            try:
                signal.pidfd_send_signal(parent_fd, signum)
            except ProcessLookupError:
                pass
        with (FOLDER / 'supervisor.lock').open('a') as original_lock:
            fcntl.flock(original_lock, fcntl.LOCK_EX)
            state['original_supervisor_suspended'] = False
            stage('ppo-2000-audit', 'audit_ppo_transfer.py', [
                '--bc', plan['objects']['mug']['checkpoint'], '--checkpoint', base.inside(checkpoint),
                '--stage', base.inside(FOLDER/'ppo-2000'), '--expected-update', 2000,
                '--out', base.inside(FOLDER/'ppo-2000-audit.json')], 'PPO_TRANSFER_AUDIT_PASSED')
            stage('evaluation-2000', 'evaluate_kitchen.py', [
                '--headless', '--config', base.inside(FOLDER/'ppo.yaml'),
                '--out', base.inside(FOLDER/'evaluation-2000'), '--num_envs', 80,
                '--checkpoint', base.inside(checkpoint), '--require-strict-success', '--seed', 12345],
                'EVALUATION_DONE')
            evaluation = base.read(FOLDER/'evaluation-2000/summary.json')
            assert evaluation['status'] == 'complete' and evaluation['completed'] == 80
            assert evaluation['require_strict_success']
            assert evaluation['evaluation_cases_sha256'] == base.sha(FOLDER/'evaluation-cases.json')
            assert evaluation['checkpoint_sha256'] == base.sha(checkpoint)
            assert all(x['strict_success_verified'] for x in evaluation['results'] if x['reason'] == 'success')
            evaluations = base.read(FOLDER/'evaluations.json')
            assert not any(x['update'] == 2000 for x in evaluations)
            evaluations.append(dict(update=2000, checkpoint=base.inside(checkpoint), cases=80,
                success_rate=evaluation['success_rate'], counts=evaluation['counts']))
            base.write(FOLDER/'evaluations.json', evaluations)
            assert not (FOLDER/'ppo-2500').exists()
            state.update(status='paused', pause_reason='User requested stop after update 2000 and its evaluation',
                         stopped_at_update=2000, updated_unix=time.time())
            base.write(FOLDER/'supervisor.json', state)
        request.update(status='complete', stopped_at_update=2000, evaluation=evaluations[-1],
                       checkpoint_sha256=base.sha(checkpoint), no_later_segment_started=True,
                       finished_unix=time.time())
        base.write(OUT/'result.json', request)
    except BaseException as exc:
        request.update(status='failed', error=repr(exc), updated_unix=time.time())
        state.update(status='stop_after_2000_failed', error=repr(exc))
        base.write(FOLDER/'supervisor.json', state)
        raise
    finally:
        base.write(OUT/'request.json', request)
        os.close(parent_fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['inspect', 'run'])
    args = parser.parse_args()
    if args.mode == 'inspect':
        print(json.dumps(inspect()[1], indent=2))
    else:
        run()
