"""Derive a left-counter-only cup experiment without changing historical inputs."""
import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import random

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inside(path):
    return '/workspace/' + str(path.resolve().relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/cup4-preppo')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/cup4-left-preppo')
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    config = yaml.safe_load((source/'task.yaml').read_text())
    prep = read(source/'preparation.json')
    assert sha(source/'task.yaml') == prep['task_sha256']
    settings = config['garmi_task']['kitchen']
    original_mask = ROOT/settings['mask'].removeprefix('/workspace/')
    assert sha(original_mask) == settings['files_sha256']['mask']
    mask = read(original_mask)
    for relative, expected in mask['sources_sha256'].items():
        assert sha(ROOT/relative) == expected, relative
    mask['regions'] = [r for r in mask['regions'] if r['name'] == 'left']
    assert len(mask['regions']) == 1
    mask['regions'][0]['cell_offset'] = 0
    mask['cells'] = [c for c in mask['cells'] if c['region'] == 'left']
    assert len(mask['cells']) == math.prod(mask['regions'][0]['shape_xy'])
    mask['derivation'] = dict(source=inside(original_mask), source_sha256=sha(original_mask),
                              operation='Keep certified left cells and their reference lists unchanged')
    settings['object_regions'] = [r for r in settings['object_regions'] if r['name'] == 'left']
    assert len(settings['object_regions']) == 1
    settings['object_regions'][0]['probability'] = 1.0
    out.mkdir(parents=True)
    write(out/'position-mask.json', mask)
    settings['mask'] = inside(out/'position-mask.json')
    settings['files_sha256']['mask'] = sha(out/'position-mask.json')
    seen = set()
    details = {}
    for filename, count, seed in [('evaluation-cases.json', 100, 2026102011),
                                   ('expert-audit-cases.json', 20, 2026102012)]:
        original = read(source/filename)
        cases = [copy.deepcopy(c) for c in original if c['region'] == 'left']
        preserved = len(cases)
        for case in cases:
            key = tuple(case['robot_state'] + case['object_state'])
            assert key not in seen
            seen.add(key)
        rng = random.Random(seed)
        for j, ref in enumerate(mask['reference_ids']):
            template = next(c for c in cases if c['reference'] == ref)
            pool = [c for c in mask['cells'] if j in c['allowed_reference_indices']]
            missing = count - sum(c['reference'] == ref for c in cases)
            assert missing >= 0 and pool
            for k in range(missing):
                lo, hi = rng.choice(pool)['bounds_map_xy']
                x, y = [rng.uniform(lo[a]+1.e-5, hi[a]-1.e-5) for a in range(2)]
                blo, bhi = settings['base_bounds_map_xy']
                bx, by = [rng.uniform(blo[a], bhi[a]) for a in range(2)]
                case = copy.deepcopy(template)
                case.update(id=f"left-{Path(filename).stem}-{ref}-{k:03}",
                            depth='front' if y < 7.363333 else 'rear' if y > 7.496667 else 'middle')
                case['object_state'][:2] = [y-6.35, .3-x]
                case['robot_state'][:2] = [by-6.35, .3-bx]
                key = tuple(case['robot_state'] + case['object_state'])
                assert key not in seen
                seen.add(key)
                cases.append(case)
        rng.shuffle(cases)
        assert collections.Counter(c['reference'] for c in cases) == dict.fromkeys(mask['reference_ids'], count)
        write(out/filename, cases)
        details[filename] = dict(preserved_left_cases=preserved, new_left_cases=len(cases)-preserved,
                                 total=len(cases), sha256=sha(out/filename))
    config['evaluation'] = dict(cases=inside(out/'evaluation-cases.json'),
                               cases_sha256=sha(out/'evaluation-cases.json'))
    config['agent']['experiment'].update(directory=inside(out), experiment_name='cup_left_preppo')
    (out/'task.yaml').write_text(yaml.safe_dump(config))
    prep.update(task_sha256=sha(out/'task.yaml'), placement_regions=['left'],
                source_experiment=inside(source), case_derivation=details,
                allowed_references_min=min(len(c['allowed_reference_indices']) for c in mask['cells']))
    write(out/'preparation.json', prep)
    print('CUP_LEFT_PREPARED', json.dumps(details), flush=True)


if __name__ == '__main__':
    main()
