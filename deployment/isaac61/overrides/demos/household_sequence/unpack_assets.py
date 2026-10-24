"""Verify and install the household task assets into the runtime workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def unpack(root, archive, manifest):
    data=json.loads(manifest.read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=data['archive_sha256']:
        raise ValueError('Household asset archive checksum mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(data['files']) or {m.name for m in members}!=set(data['files']):
            raise ValueError('Household asset archive members differ from manifest')
        for m in members:
            if not m.isfile() or not (root/m.name).resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe household asset member')
            if hashlib.sha256(tar.extractfile(m).read()).hexdigest()!=data['files'][m.name]:
                raise ValueError('Household asset checksum mismatch: '+m.name)
        tar.extractall(root,filter='data')
    # Shared frozen resources for the isolated household policy module tree.
    for path,target in [('household_rl/assets',root/'rl/assets')]:
        link=root/path;link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(target, target_is_directory=True)
    print('HOUSEHOLD_ASSETS_VERIFIED',len(members))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();unpack(a.root,a.archive,a.manifest)
