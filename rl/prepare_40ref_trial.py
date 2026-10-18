"""Freeze the 40-reference task, separate audit cases and 4000 evaluation cases."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sample_cases(config, mask, references, n, seed, prefix, forbidden=None):
    forbidden = set() if forbidden is None else forbidden
    rng = random.Random(seed)
    cases = []
    for j, ref in enumerate(references):
        cells = {region: [c for c in mask['cells'] if c['region'] == region
                         and j in c['allowed_reference_indices']] for region in ('left', 'right')}
        if not cells['left'] or not cells['right']:
            raise ValueError(f'{ref} has no allowed cell in one region')
        for k in range(n):
            region = 'right' if k >= round(n * .75) else 'left'
            lo, hi = rng.choice(cells[region])['bounds_map_xy']
            x, y = [rng.uniform(lo[a] + 1.e-5, hi[a] - 1.e-5) for a in range(2)]
            base_lo, base_hi = config['garmi_task']['kitchen']['base_bounds_map_xy']
            bx, by = [rng.uniform(base_lo[a], base_hi[a]) for a in range(2)]
            case = dict(object='bowl', reference=ref, id=f'{prefix}-{ref}-{k:03d}',
                robot_state=[by-6.35, .3-bx, .0259, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                object_state=[y-6.35, .3-x, mask['bowl_origin_z_m'], 2.**-.5, 0., 0., -2.**-.5,
                              0., 0., 0., 0., 0., 0.], bowl_map_xy=[x, y], region=region,
                depth='front' if y < 7.363333 else 'rear' if y > 7.496667 else 'middle')
            key = tuple(case['robot_state'] + case['object_state'])
            assert key not in forbidden
            forbidden.add(key)
            cases.append(case)
    rng.shuffle(cases)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/bowl40')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Use the optional imported four-reference checkpoint/data')
    args = parser.parse_args()
    out = args.out.resolve()
    out.relative_to(ROOT)
    cfg = yaml.safe_load((ROOT / 'configs/task.yaml').read_text())
    manifest = json.loads((ROOT / 'assets/manifest.json').read_text())
    for relative, expected in manifest['files_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    bank = json.loads((ROOT / 'assets/grasp_bank/selected-40.json').read_text())
    mask = json.loads((ROOT / 'assets/reference_filter/position-mask.json').read_text())
    refs = [r['id'] for r in bank['objects'][0]['references']]
    assert len(refs) == 40 and refs == mask['reference_ids']
    assert refs == cfg['garmi_task']['multi_object']['reference_ids']['bowl']
    oldrefs, checkpoint, checkpoint_sha, forbidden = [], None, None, set()
    if args.bootstrap:
        imported = json.loads((ROOT / 'bootstrap/manifest.json').read_text())
        for relative, expected in imported['files_sha256'].items():
            assert sha(ROOT / relative) == expected, relative
        data = json.loads((ROOT / 'bootstrap/base-train-index.json').read_text())
        oldrefs = sorted({r['reference'] for r in data['demonstrations']})
        assert len(oldrefs) == 4 and set(oldrefs) <= set(refs)
        checkpoint = '/workspace/bootstrap/initial.pt'
        checkpoint_sha = sha(ROOT / 'bootstrap/initial.pt')
        original = json.loads((ROOT / 'bootstrap/regression-cases.json').read_text())
        forbidden = {tuple(c['robot_state'] + c['object_state']) for c in original}
    tests = sample_cases(cfg, mask, refs, 100, 2026101801, 'eval40', forbidden)
    audit = sample_cases(cfg, mask, refs, 10, 2026101802, 'audit40', forbidden)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'evaluation-cases-4000.json', tests)
    save(out / 'expert-check-cases-400.json', audit)
    cfg['evaluation'] = dict(cases='/workspace/' + str((out / 'evaluation-cases-4000.json').relative_to(ROOT)),
        cases_sha256=sha(out / 'evaluation-cases-4000.json'))
    cfg['garmi_task']['kitchen']['files_sha256'] = {
        'mask': sha(ROOT / 'assets/reference_filter/position-mask.json'),
        'scene_usd': sha(ROOT / 'assets/kitchen/kitchen.usda')}
    (out / 'task.yaml').write_text(yaml.safe_dump(cfg))
    save(out / 'preparation.json', dict(status='passed', references=refs, old_references=oldrefs,
        new_references=[r for r in refs if r not in oldrefs], checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha, bootstrap=args.bootstrap,
        ppo_updates=8000, ppo_num_envs=4096, rollout=24))
    print('PREPARED_40REF', out)


if __name__ == '__main__':
    main()
