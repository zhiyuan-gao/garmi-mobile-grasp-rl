"""Export/verify the exact data needed to resume mug PPO after update 2000.

Code is distributed through Git. Assets and expert data retain their original
/workspace-relative paths and hashes; no histories or datasets are rewritten.
"""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

import yaml

MANIFEST = 'transfer/mug-ppo-2000/manifest.json'
RUN = Path('runs/wrist-ppo-v6/mug')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def workspace_relative(value):
    path = Path(value)
    if path.is_absolute():
        path = path.relative_to('/workspace')
    if '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe workspace path: {value}')
    return path


def export(source, code_root, output):
    state = json.loads((source / RUN / 'supervisor.json').read_text())
    assert state['status'] == 'paused' and state['ppo_updates'] == 2000
    stopped = json.loads((source / RUN / 'stop-after-2000/result.json').read_text())
    assert stopped['status'] == 'complete' and stopped['stopped_at_update'] == 2000
    assert not (source / RUN / 'ppo-2500').exists()
    output.mkdir(parents=True, exist_ok=False)
    profile = yaml.safe_load((source / RUN / 'ppo.yaml').read_text())
    selected = {}

    def add(relative, expected=None):
        relative = workspace_relative(relative)
        path = source / relative
        assert path.is_file(), relative
        assert path.resolve().is_relative_to(source), relative
        digest = sha(path)
        if expected is not None:
            assert digest == expected, f'Input hash mismatch: {relative}'
        selected[str(relative)] = dict(sha256=digest, bytes=path.stat().st_size)

    # Complete small asset trees include USD sublayers, robot URDF, collision
    # meshes and repaired mug geometry. Resource binaries are not all in Git.
    for directory in ('assets', 'rl/assets', 'resources'):
        for path in sorted((source / directory).rglob('*')):
            if path.is_file() and path.suffix != '.md' and '__pycache__' not in path.parts:
                add(path.relative_to(source))
    mask_path = workspace_relative(profile['garmi_task']['kitchen']['mask'])
    mask = json.loads((source / mask_path).read_text())
    for relative, digest in mask['sources_sha256'].items():
        add(relative, digest)
    for name in ('ppo.yaml', 'evaluation-cases.json', 'training-reset-pool.json',
                 'split-audit.json', 'evaluations.json', 'ppo-2000-audit.json',
                 'evaluation-2000/summary.json', 'stop-after-2000/result.json'):
        add(RUN / name)
    for name in ('summary.json', 'progress.json', 'latest.json', 'task.json',
                 'reward_terms.json', 'expert_data.json', 'initialization_1500.json',
                 'checkpoints/update_2000.pt', 'checkpoints/update_2000.resume.pt'):
        add(RUN / 'ppo-2000' / name)
    imitation = Path('runs/wrist-imitation-v6/mug')
    add(imitation / 'round-03-bc/last.pt')
    add(imitation / 'result.json')
    index = workspace_relative(profile['expert_bc']['index'])
    add(index)
    data = json.loads((source / index).read_text())
    trajectories = len(data['demonstrations'])
    dagger_files = len(data['dagger_label_sets'])
    for row in data['demonstrations'] + data['dagger_label_sets']:
        add(row['file'], row['sha256'])
    del data
    gc.collect()
    pool_spec = profile['garmi_task']['multi_object']['training_reset_pool']
    add(pool_spec['file'], pool_spec['sha256'])
    add(profile['evaluation']['cases'], profile['evaluation']['cases_sha256'])
    # Include reset-pool provenance, though runtime samples the frozen pool itself.
    pool = json.loads((source / workspace_relative(pool_spec['file'])).read_text())
    add(pool['source_index'], pool['source_sha256'])
    code_files = list((source / 'rl').rglob('*.py')) + [
        source / 'scripts/rl.sh', source / 'scripts/container_exec.sh']
    code_hashes = {}
    for path in code_files:
        relative = path.relative_to(source)
        digest = sha(path)
        assert sha(code_root / relative) == digest, f'Published runtime differs: {relative}'
        code_hashes[str(relative)] = digest
    checkpoint = RUN / 'ppo-2000/checkpoints/update_2000.pt'
    assert selected[str(checkpoint)]['sha256'] == stopped['checkpoint_sha256']
    manifest = dict(schema_version=1, object='mug', update=2000, created_unix=time.time(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code_root, text=True).strip(),
        code_branch='codex/ycb-wrist-v6-ppo2000', code_sha256=code_hashes,
        checkpoint='/workspace/' + str(checkpoint), config='/workspace/' + str(RUN/'ppo.yaml'),
        original_num_envs=4096, seed=2026102002,
        expert_trajectories=trajectories, dagger_label_files=dagger_files,
        strict_evaluation=stopped['evaluation'], files=selected,
        total_uncompressed_bytes=sum(x['bytes'] for x in selected.values()),
        resume_semantics='Restore optimizer, scheduler, normalizers and RNG; start fresh simulation episodes. Different hardware or num_envs is not bitwise identical.')
    raw = (json.dumps(manifest, indent=2) + '\n').encode()
    (output / 'manifest.json').write_bytes(raw)
    archive = output / 'mug-ppo-2000-resume.tar.gz'
    partial = archive.with_suffix(archive.suffix + '.part')
    print('PACKING', len(selected), 'files', manifest['total_uncompressed_bytes'], 'bytes', flush=True)
    with tarfile.open(partial, 'w:gz', compresslevel=3, dereference=True) as tar:
        for relative in sorted(selected):
            tar.add(source / relative, arcname=relative, recursive=False)
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    partial.replace(archive)
    (output / 'SHA256SUMS').write_text(''.join(
        f'{sha(path)}  {path.name}\n' for path in [archive, output/'manifest.json']))
    print('RESUME_BUNDLE_EXPORTED', archive, archive.stat().st_size, flush=True)


def verify(root):
    manifest = json.loads((root / MANIFEST).read_text())
    for relative, entry in manifest['files'].items():
        path = root / workspace_relative(relative)
        assert path.stat().st_size == entry['bytes'], relative
        assert sha(path) == entry['sha256'], relative
    for relative, digest in manifest['code_sha256'].items():
        assert sha(root / workspace_relative(relative)) == digest, relative
    print('RESUME_BUNDLE_VERIFIED', len(manifest['files']), 'files, update', manifest['update'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--source', type=Path, required=True)
    e.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1])
    e.add_argument('--out', type=Path, required=True)
    v = sub.add_parser('verify')
    v.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    if a.command == 'export':
        export(a.source.resolve(), a.code_root.resolve(), a.out.resolve())
    else:
        verify(a.root.resolve())
