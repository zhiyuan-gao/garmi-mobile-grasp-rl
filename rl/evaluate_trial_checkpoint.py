"""Run fixed trial evaluations after the training GPU process has exited."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import traceback

import torch
import yaml
from garmi_rl.trial_evaluation import atomic_json, evaluate_checkpoint, evaluate_paired_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--eval-envs', type=int, default=240)
parser.add_argument('--eval-seed', type=int, default=12345)
parser.add_argument('--paired-eval-cases')
parser.add_argument('--paired-eval-envs', type=int, default=120)
parser.add_argument('--paired-eval-seed', type=int, default=20261018)
args = parser.parse_args()
out = Path(args.out)
checkpoint = Path(args.checkpoint)
state_path = checkpoint.with_suffix('.resume.pt')
state = torch.load(state_path, map_location='cpu', weights_only=False)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == state['checkpoint_sha256']
progress = json.loads((out / 'progress.json').read_text())
assert progress['update'] == state['update']
agent_cfg = yaml.safe_load((out / 'ppo.yaml').read_text())
progress.update(pid=os.getpid(), evaluation_mode='after_training_process_exit')


def stop(signum, frame):
    raise KeyboardInterrupt('Deferred evaluation interrupted')


signal.signal(signal.SIGTERM, stop)
try:
    kw = dict(out=out, args=args, agent_cfg=agent_cfg, progress=progress)
    evaluate_checkpoint(checkpoint, state['update'], **kw)
    if args.paired_eval_cases:
        evaluate_paired_checkpoint(checkpoint, state['update'], **kw)
    progress['status'] = 'complete'
    state['progress'] = progress
    temporary = state_path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(state_path)
    atomic_json(out / 'progress.json', progress)
    atomic_json(out / 'summary.json', progress)
    print('DEFERRED_EVALUATION_DONE', flush=True)
except BaseException as exc:
    progress.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                    error=repr(exc), traceback=traceback.format_exc())
    atomic_json(out / 'progress.json', progress)
    raise
