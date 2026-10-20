"""Verify original policy assets and an explicitly recorded integrated runtime."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_workspace_files(manifest_path, workspace):
    workspace = Path(workspace)
    manifest = json.loads(Path(manifest_path).read_text())
    compatibility_path = workspace / 'runtime/code_compatibility.json'
    code = {}
    if compatibility_path.exists():
        compatibility = json.loads(compatibility_path.read_text())
        if compatibility['original_manifest_sha256'] == sha(manifest_path):
            code = compatibility['code_sha256']
            for name, digest in code.items():
                if not name.startswith('rl/garmi_rl/') or not name.endswith('.py') or '..' in Path(name).parts:
                    raise ValueError(f'Runtime compatibility cannot replace an asset: {name}')
                if sha(workspace / name) != digest:
                    raise ValueError(f'Integrated runtime changed: {name}')
    for name, digest in manifest['workspace_files'].items():
        relative = name.removeprefix('/workspace/')
        expected = code.get(relative, digest)
        if sha(workspace / relative) != expected:
            raise ValueError(f'Training resource changed: {name}')
