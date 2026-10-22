"""Snapshot the live RL demo for the original CRAM Lab image build."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'integration/native-binder/overlay/demos/garmi_rl'


def payload_files(root):
    files = set()
    for folder in ('integration/garmi_cram_rl', 'integration/tests', 'rl/garmi_rl',
                   'runtime/cram_vrb_lab', 'scripts'):
        files.update(p for p in (root / folder).rglob('*.py') if '__pycache__' not in p.parts)
    files.update((root / 'integration').glob('*.py'))
    files.update(p for p in (root / 'integration/web').rglob('*') if p.is_file())
    files.update(root / name for name in (
        'integration/fastdds.xml', 'integration/garmi-desktop.kit',
        'integration/GARMI_RL_pick_up.desktop', 'integration/left_counter_episodes.json',
        'runtime/unpack.py', 'runtime/data.tar.gz', 'runtime/data-manifest.json',
        'scripts/setup_isaac61.sh', 'scripts/local_python.sh', 'scripts/local_python61.sh',
        'scripts/isaaclab23-sim61.patch', 'scripts/requirements-isaac61.lock.txt',
        'rl/build-constraints.txt', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md',
        'cram-vrb-lab/assets/garmi-apartment/world.usda'))
    return sorted(files)


def build_payload(root=ROOT, destination=DESTINATION):
    # Fail before packaging if a frozen training input has drifted.
    policy = json.loads((root / 'results/cram-rl-integration/policy-7500/manifest.json').read_text())
    for name, digest in policy['workspace_files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen policy dependency mismatch: {name}')
    files = payload_files(root)
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'runtime.tar.gz'
    with path.open('wb') as output, gzip.GzipFile(fileobj=output, mode='wb', mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for source in files:
                data = source.read_bytes()
                member = tarfile.TarInfo(str(source.relative_to(root)))
                member.size, member.mode, member.mtime = len(data), 0o644, 0
                archive.addfile(member, io.BytesIO(data))
    (destination / 'files.sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Packaged {len(files)} files into {path} ({path.stat().st_size / 1024**2:.1f} MiB).')
    return path


if __name__ == '__main__':
    build_payload()
