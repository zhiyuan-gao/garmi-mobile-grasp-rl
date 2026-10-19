"""Install the explicitly selected runtime data and verify the frozen input contract."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path('/workspace')
runtime = root / 'runtime'
expected = json.loads((runtime / 'data-manifest.json').read_text())
with tarfile.open(runtime / 'data.tar.gz', 'r:gz') as archive:
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if not target.is_relative_to(root) or member.issym() or member.islnk():
            raise ValueError(f'Invalid runtime archive member: {member.name}')
    archive.extractall(root, filter='data')
for name, digest in expected.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f'Runtime asset mismatch: {name}')
manifest = json.loads((root / 'results/cram-rl-integration/policy-7500/manifest.json').read_text())
for name, digest in manifest['workspace_files'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise ValueError(f'Frozen policy dependency mismatch: {name}')
print(f'Validated {len(expected)} runtime files and {len(manifest["workspace_files"])} policy dependencies.')
