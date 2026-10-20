"""Transfer checks reject missing/corrupt payloads and unsafe relative paths."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('resume_bundle',
    Path(__file__).resolve().parents[1] / 'scripts/export_ppo_resume.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


@pytest.mark.parametrize('path', ['../outside', '/home/user/file', '/workspace/../outside'])
def test_refuse_non_workspace_paths(path):
    with pytest.raises(ValueError):
        bundle.workspace_relative(path)


@pytest.mark.parametrize('damage', ['payload', 'code', 'missing'])
def test_verify_rejects_changed_or_missing_files(tmp_path, damage):
    payload = tmp_path / 'data.bin'
    code = tmp_path / 'train.py'
    payload.write_bytes(b'checkpoint')
    code.write_text('original runtime')
    manifest = tmp_path / bundle.MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(update=2000,
        files={'data.bin': dict(bytes=payload.stat().st_size, sha256=bundle.sha(payload))},
        code_sha256={'train.py': bundle.sha(code)})))
    bundle.verify(tmp_path)
    if damage == 'payload':
        payload.write_bytes(b'CHECKPOINT')
    elif damage == 'code':
        code.write_text('modified runtime')
    else:
        payload.unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        bundle.verify(tmp_path)
