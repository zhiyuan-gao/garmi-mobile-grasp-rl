import hashlib
import json
import pytest
from garmi_cram_rl.resources import verify_workspace_files


def sha(data):
    return hashlib.sha256(data).hexdigest()


def setup_bundle(root):
    code = root / 'rl/garmi_rl/example.py'
    code.parent.mkdir(parents=True)
    code.write_bytes(b'new runtime')
    asset = root / 'asset.usda'
    asset.write_bytes(b'original asset')
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'workspace_files': {
        'rl/garmi_rl/example.py': sha(b'old runtime'), 'asset.usda': sha(asset.read_bytes())}}))
    runtime = root / 'runtime'
    runtime.mkdir()
    (runtime / 'code_compatibility.json').write_text(json.dumps({
        'original_manifest_sha256': sha(manifest.read_bytes()),
        'code_sha256': {'rl/garmi_rl/example.py': sha(code.read_bytes())}}))
    return manifest, code, asset


def test_explicit_runtime_migration_keeps_original_manifest(tmp_path):
    manifest, _, _ = setup_bundle(tmp_path)
    original = manifest.read_bytes()
    verify_workspace_files(manifest, tmp_path)
    assert manifest.read_bytes() == original


@pytest.mark.parametrize('target', ['code', 'asset', 'manifest'])
def test_changed_code_assets_or_provenance_are_rejected(tmp_path, target):
    manifest, code, asset = setup_bundle(tmp_path)
    if target == 'manifest':
        d = json.loads(manifest.read_text())
        d['unknown_policy'] = True
        manifest.write_text(json.dumps(d))
    else:
        (code if target == 'code' else asset).write_bytes(b'changed')
    with pytest.raises(ValueError):
        verify_workspace_files(manifest, tmp_path)
