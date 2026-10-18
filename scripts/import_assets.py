"""Import the exact validated assets from an existing FAME_RL workspace.

No geometry is regenerated. Check original hashes before relocating metadata;
retain original provenance and regenerate the dependent bank/mask hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def import_assets(source, destination=ROOT):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    mapping = json.loads((ROOT / 'resources/asset-sources.json').read_text())
    # Validate the entire input before writing any output.
    for old, info in mapping.items():
        if sha(source / old) != info['sha256']:
            raise ValueError(f'Validated asset changed: {old}')
    for old, info in mapping.items():
        target = destination / info['target']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / old, target)
    bank_path = destination / 'assets/grasp_bank/selected-40.json'
    bank = json.loads(bank_path.read_text())
    bowl = bank['objects'][0]
    bank = {'objects': [{k: bowl[k] for k in
        ('name', 'mass', 'bounds', 'collider', 'mesh_sha256', 'usd_sha256')}],
        'provenance': {'original_sha256': sha(source / 'results/rl-preflight/bowl-kitchen-40refs/selected-40.json')}}
    bank['objects'][0].update(mesh='/workspace/assets/bowl.stl',
        usd='/workspace/rl/assets/bowl.usda',
        references=json.loads((ROOT / 'resources/bowl-40-references.json').read_text()))
    original = [{k: r[k] for k in ('id', 'rank', 'score', 'mesh_T_gripper', 'position', 'quaternion')}
                for r in bowl['references']]
    if original != bank['objects'][0]['references']:
        raise ValueError('Published reference definitions differ from validated bank')
    save(bank_path, bank)
    mask_path = destination / 'assets/reference_filter/position-mask.json'
    mask = json.loads(mask_path.read_text())
    original_hashes = mask['sources_sha256'].copy()
    mask['bank_path'] = mapping[mask['bank_path']]['target']
    mask['sources_sha256'] = {mapping[key]['target']: sha(destination / mapping[key]['target'])
                              for key in original_hashes}
    mask['relocation_provenance'] = {'original_sources_sha256': original_hashes,
        'note': 'Only paths and bank metadata changed; cells, indices and geometry are unchanged.'}
    # The old validation report is not bundled; retain its hash as provenance.
    mask.pop('validation_file', None)
    save(mask_path, mask)
    inputs = {info['target']: sha(destination / info['target']) for info in mapping.values()}
    save(destination / 'assets/manifest.json', {'status': 'imported', 'files_sha256': inputs})
    return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-workspace', required=True, type=Path)
    args = parser.parse_args()
    imported = import_assets(args.from_workspace)
    print(f'ASSETS_IMPORTED: {len(imported)} verified files; binaries remain ignored by Git')
