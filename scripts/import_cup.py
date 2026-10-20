"""Copy the previously verified cup asset and grasp definitions without modifying physics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from-workspace', type=Path, required=True)
    args = p.parse_args()
    source = args.from_workspace.resolve()
    bank = json.loads((source/'rl/assets/multi/validated.json').read_text())
    cup = copy.deepcopy(next(o for o in bank['objects'] if o['name'] == 'cup'))
    dest = ROOT/'assets/cup'
    dest.mkdir(parents=True, exist_ok=False)
    provenance = {}
    for key, target in [('mesh', 'cup.stl'), ('usd', 'cup.usda')]:
        old = source/cup[key].removeprefix('/workspace/')
        assert sha(old) == cup[key+'_sha256'], key
        shutil.copy2(old, dest/target)
        provenance[str(old.relative_to(source))] = sha(old)
        cup[key] = '/workspace/assets/cup/'+target
    old = source/'results/rl-trials/trial-05-diagnosis/cup-hulls.json'
    shutil.copy2(old, dest/'hulls.json')
    provenance[str(old.relative_to(source))] = sha(old)
    cup.pop('screening', None)
    (dest/'bank.json').write_text(json.dumps({'objects': [cup]}, indent=2)+'\n')
    for name in ('summary.json', 'cup-cooked.json', 'asset-audit.json'):
        old = source/'results/rl-trials/trial-05-diagnosis'/name
        shutil.copy2(old, dest/('previous-'+name))
        provenance[str(old.relative_to(source))] = sha(old)
    (dest/'import.json').write_text(json.dumps(dict(files_sha256=provenance,
        object='cup', geometry_rescaled=False, physics_changed=False), indent=2)+'\n')
    print('CUP_IMPORTED', len(cup['references']), 'previously physically verified references')


if __name__ == '__main__':
    main()
