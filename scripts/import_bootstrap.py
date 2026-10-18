"""Optionally import the exact prior checkpoint/data used by the current run.

Only referenced files are copied, never the complete historical experiment.
All imported artifacts stay in the Git-ignored bootstrap directory.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = 'results/rl-trials/trial-16-gaddpg-augmentation'
CHECKPOINT_SHA = '7b4131d7a0659402c767d09379f323db00a8e99fd68f2e381d7f13b80990d6a9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_bootstrap(source):
    source = Path(source).resolve()
    out = ROOT / 'bootstrap'
    if (out / 'manifest.json').exists():
        raise ValueError('Bootstrap already imported; use the existing verified copy')
    (out / 'data').mkdir(parents=True, exist_ok=True)
    saved = {}

    def payload(row):
        relative = Path(row['file']).relative_to('/workspace')
        path = source / relative
        digest = sha(path)
        if digest != row['sha256']:
            raise ValueError(f'Data hash mismatch: {relative}')
        target = out / 'data' / (digest + path.suffix)
        if not target.exists():
            shutil.copy2(path, target)
        row['file'] = '/workspace/' + str(target.relative_to(ROOT))
        saved[str(target.relative_to(ROOT))] = digest

    for name in ('base-train-index', 'base-validation-index', 'round-03-train-index', 'round-03-validation-index'):
        data = copy.deepcopy(json.loads((source / SOURCE_RUN / (name + '.json')).read_text()))
        for row in data.get('demonstrations', []) + data.get('dagger_label_sets', []):
            payload(row)
        # Remove run-log provenance, retaining hashes and actual sample metadata.
        for key in ('source_indices', 'source_checkpoint', 'checkpoint', 'index'):
            data.pop(key, None)
        path = out / (name + '.json')
        path.write_text(json.dumps(data, indent=2) + '\n')
        saved[str(path.relative_to(ROOT))] = sha(path)
    checkpoint = source / SOURCE_RUN / 'ppo-2000-4096/stage-1500/checkpoints/update_1500.pt'
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Incorrect initialization checkpoint')
    shutil.copy2(checkpoint, out / 'initial.pt')
    shutil.copy2(source / 'results/rl-trials/trial-11-bowl-kitchen-4refs/evaluation-cases.json',
                 out / 'regression-cases.json')
    for name in ('initial.pt', 'regression-cases.json'):
        saved['bootstrap/' + name] = sha(out / name)
    (out / 'manifest.json').write_text(json.dumps({'files_sha256': saved}, indent=2) + '\n')
    print('BOOTSTRAP_IMPORTED', len(saved), 'files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-workspace', required=True, type=Path)
    import_bootstrap(parser.parse_args().from_workspace)
