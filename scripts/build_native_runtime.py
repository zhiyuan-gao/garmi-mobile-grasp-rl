"""Verify and package the versioned Isaac Sim 6.1 deployment sources (CPU only)."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

MAP = 'deployment/isaac61/source-map.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Invalid relative path: {value}')
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves source tree: {value}')
    return result


def verify(root):
    data = json.loads((root / MAP).read_text())
    for target, entry in data['files'].items():
        safe_path(root, target)
        source = safe_path(root, entry['source'])
        if not source.is_file() or digest(source) != entry['sha256']:
            raise ValueError(f'Source hash mismatch: {entry["source"]}')
    return data


def build(root, out):
    data = verify(root)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root).strip():
        raise ValueError('Commit source changes before building a release')
    for entry in data['files'].values():
        subprocess.check_call(['git', 'ls-files', '--error-unmatch', entry['source']], cwd=root,
                              stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    out.mkdir(parents=True, exist_ok=False)
    archive = out / 'runtime.tar.gz'
    with archive.open('wb') as f, gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for name, entry in sorted(data['files'].items()):
                source = root / entry['source']
                info = tarfile.TarInfo(name)
                info.size = source.stat().st_size
                info.mode = entry['mode']
                with source.open('rb') as reader:
                    tar.addfile(info, reader)
    hashes = {name: entry['sha256'] for name, entry in data['files'].items()}
    manifest = out / 'files.sha256.json'
    manifest.write_text(json.dumps(hashes, indent=2) + '\n')
    provenance = dict(schema_version=1,
        repository='https://github.com/zhiyuan-gao/garmi-mobile-grasp-rl', commit=commit,
        source_map=MAP, source_map_sha256=digest(root / MAP),
        archive_sha256=digest(archive), files_manifest_sha256=digest(manifest),
        file_count=len(hashes), environment=data['environment'],
        content_baseline=data['imported_from'], task='bowl_counter_to_table')
    (out / 'source.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('NATIVE_RUNTIME_BUILT', out, commit, len(hashes))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=['verify', 'build'])
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.command == 'verify':
        print('NATIVE_SOURCES_VERIFIED', len(verify(args.root.resolve())['files']))
    else:
        if args.out is None:
            p.error('build requires --out')
        build(args.root.resolve(), args.out.resolve())
