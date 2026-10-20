"""Select four distinct verified grasps and certify their kitchen target geometry."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import random

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from cup_goal_geometry import GoalGeometry

ROOT = Path(__file__).resolve().parents[1]
CUP = ROOT/'assets/cup'
OUT = ROOT/'runs/cup4-preppo'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    bank = json.loads((CUP/'bank.json').read_text())
    obj = bank['objects'][0]
    # Retain only references that passed all ten prior physical finish tests.
    refs = [r for r in obj['references'] if r['confirmation_counts'] == {'success': 10}]
    obj['references'] = refs
    write(CUP/'bank.json', bank)
    geometry = GoalGeometry()
    config = yaml.safe_load((ROOT/'configs/task.yaml').read_text())
    settings = config['garmi_task']['kitchen']
    regions = settings.pop('bowl_regions')
    settings.update(object_name='cup', object_regions=regions,
                    object_map_yaw_degrees=settings.pop('bowl_map_yaw_degrees'))
    cells, region_meta = [], []
    for region in regions:
        lo, hi = np.asarray(region['bounds_map_xy'])
        shape = np.rint((hi-lo)/.02).astype(int)
        offset = len(cells)
        for ix, iy in itertools.product(range(shape[0]), range(shape[1])):
            lower = lo+np.array([ix, iy])*.02
            cells.append(dict(region=region['name'], index_xy=[ix, iy],
                bounds_map_xy=[lower.tolist(), (lower+.02).tolist()], allowed_reference_indices=[]))
        region_meta.append(dict(name=region['name'], bounds_map_xy=region['bounds_map_xy'],
                                shape_xy=shape.tolist(), cell_offset=offset))
    clear = np.zeros((len(cells), len(refs)), dtype=bool)
    for j, ref in enumerate(refs):
        swept = geometry.query(j, (.01, .01), .002)
        for i, cell in enumerate(cells):
            clear[i, j] = not swept.collides(np.mean(cell['bounds_map_xy'], axis=0))
        print('CUP_GEOMETRY', ref['id'], int(clear[:, j].sum()), '/', len(cells), flush=True)
    rotations = [Rotation.from_quat(np.array(r['quaternion'])[[1, 2, 3, 0]]) for r in refs]
    midpoint = np.array([np.array(r['position'])+rot.apply([0., 0., .10365])
                         for r, rot in zip(refs, rotations)])
    def separation(a, b):
        angle = (rotations[a].inv()*rotations[b]).magnitude()
        return np.linalg.norm(midpoint[a]-midpoint[b]) + .025*angle
    anchor = next(i for i, r in enumerate(refs) if r['id'] == 'cup-r009')
    choices = [s for s in itertools.combinations(range(len(refs)), 4)
               if anchor in s and clear[:, s].any(axis=1).all()]
    if not choices:
        raise ValueError('No four-reference set covers the complete placement region')
    selected = max(choices, key=lambda s: (
        min(separation(a,b) for a,b in itertools.combinations(s,2)),
        float(clear[:,s].sum())))
    selected_refs = [refs[i] for i in selected]
    write(CUP/'selection.json', dict(references=[r['id'] for r in selected_refs],
        candidate_references=[r['id'] for r in refs], preserved_anchor='cup-r009',
        criterion='Maximize the minimum pad-midpoint separation plus 0.025 m/rad rotation separation; retain at least one valid goal in every cell',
        pairs=[dict(a=refs[a]['id'], b=refs[b]['id'],
                    midpoint_distance_m=float(np.linalg.norm(midpoint[a]-midpoint[b])),
                    angle_degrees=float(np.rad2deg((rotations[a].inv()*rotations[b]).magnitude())))
               for a,b in itertools.combinations(selected,2)]))
    obj['references'] = selected_refs
    write(CUP/'bank.json', bank)
    geometry = GoalGeometry()
    for i, cell in enumerate(cells):
        cell['allowed_reference_indices'] = [j for j, old in enumerate(selected) if clear[i, old]]
    # Independently inspect exact goals at all cell corners and two interior
    # positions, as well as original triangle meshes at an interior point.
    independent = trimesh.collision.CollisionManager()
    for record, mesh in geometry.scene_meshes:
        independent.add_object(record['path'], mesh)
    exact = [geometry.query(j) for j in range(4)]
    rng = np.random.default_rng(2026102001)
    checks = 0
    for cell in cells:
        lo, hi = np.asarray(cell['bounds_map_xy'])
        points = [[lo[0],lo[1]], [lo[0],hi[1]], [hi[0],lo[1]], [hi[0],hi[1]],
                  rng.uniform(lo,hi).tolist(), rng.uniform(lo,hi).tolist()]
        for j in cell['allowed_reference_indices']:
            for point in points:
                assert not exact[j].collides(point), (cell,j,point)
                checks += 1
            transform = np.eye(4)
            transform[:3,3] = [points[-1][1]-6.35, .3-points[-1][0], 0.]
            assert not any(independent.in_collision_single(m, transform=transform)
                           for m in geometry.reference_meshes[j])
    # Replace only Object convex parts in the exact training robot collision export.
    original = pickle.load((ROOT/'assets/kitchen/cooked-model.pkl').open('rb'))
    robot = [(record,mesh) for record,mesh in original if record['parent'] != 'Object']
    hulls = json.loads((CUP/'hulls.json').read_text())
    for index, hull in enumerate(hulls):
        robot.append((dict(parent='Object', body='Object', path=f'cup/convex_{index}'),
                      trimesh.Trimesh(vertices=hull['vertices'], faces=hull['triangles'], process=False)))
    with (CUP/'cooked-model.pkl').open('wb') as stream:
        pickle.dump(robot, stream)
    dependencies = ['assets/cup/bank.json', 'assets/cup/cup.stl', 'assets/cup/cup.usda',
                    'assets/cup/hulls.json', 'assets/cup/cooked-model.pkl',
                    'assets/kitchen/scene_meshes.pkl', 'assets/kitchen/kitchen.usda',
                    'rl/assets/manifest.json', 'rl/assets/garmi_training.urdf']
    hashes = {name:sha(ROOT/name) for name in dependencies}
    mask = dict(schema_version=1, status='validated', object_name='cup', object_map_yaw_degrees=0.,
        object_origin_z_m=.946-obj['bounds'][0][2],
        reference_ids=[r['id'] for r in selected_refs], bank_path='assets/cup/bank.json',
        cell_size_m=.02, extra_margin_m=.002, numeric_position_tolerance_m=1.e-6,
        sources_sha256=hashes, regions=region_meta, cells=cells,
        method='Same swept convex hand geometry as the bowl; 2 cm cells plus 2 mm margin',
        scope='Target hand vs closed furniture, not arm reachability or approach success',
        validation=dict(exact_corner_and_interior_checks=checks, violations=0, original_triangle_checks=True))
    write(CUP/'position-mask.json', mask)
    write(CUP/'expert.json', dict(object_name='cup', bank='assets/cup/bank.json',
        cooked_model='assets/cup/cooked-model.pkl', files_sha256=hashes))
    config.pop('expert_bc', None)
    config.pop('evaluation', None)
    config['garmi_task']['multi_object'] = dict(manifest='/workspace/assets/cup/bank.json',
        objects=['cup'], reference_ids={'cup':mask['reference_ids']}, split='all')
    settings.update(mask='/workspace/assets/cup/position-mask.json',
        files_sha256={'mask':sha(CUP/'position-mask.json'), 'scene_usd':sha(ROOT/'assets/kitchen/kitchen.usda')})
    OUT.mkdir(parents=True)
    forbidden = set()
    def sample(n, seed, prefix):
        rand = random.Random(seed)
        cases = []
        for j, ref in enumerate(mask['reference_ids']):
            for k in range(n):
                side = 'right' if k >= round(n*.75) else 'left'
                pool = [c for c in cells if c['region']==side and j in c['allowed_reference_indices']]
                assert pool, (ref,side)
                lo, hi = rand.choice(pool)['bounds_map_xy']
                x,y = [rand.uniform(lo[a]+1.e-5,hi[a]-1.e-5) for a in range(2)]
                blo,bhi = settings['base_bounds_map_xy']
                bx,by = [rand.uniform(blo[a],bhi[a]) for a in range(2)]
                case = dict(id=f'{prefix}-{ref}-{k:03}', object='cup',reference=ref,region=side,
                    depth='front' if y<7.363333 else 'rear' if y>7.496667 else 'middle',
                    robot_state=[by-6.35,.3-bx,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                    object_state=[y-6.35,.3-x,mask['object_origin_z_m'],2**-.5,0.,0.,-2**-.5,0.,0.,0.,0.,0.,0.])
                key=tuple(case['robot_state']+case['object_state'])
                assert key not in forbidden
                forbidden.add(key)
                cases.append(case)
        rand.shuffle(cases)
        return cases
    write(OUT/'evaluation-cases.json',sample(100,2026102002,'evaluation'))
    write(OUT/'expert-audit-cases.json',sample(20,2026102003,'expert-audit'))
    config['evaluation'] = dict(cases='/workspace/runs/cup4-preppo/evaluation-cases.json',
        cases_sha256=sha(OUT/'evaluation-cases.json'))
    config['agent']['experiment']['directory'] = '/workspace/runs/cup4-preppo'
    config['agent']['experiment']['experiment_name'] = 'cup_preppo'
    (OUT/'task.yaml').write_text(yaml.safe_dump(config))
    write(OUT/'preparation.json', dict(status='prepared', object='cup',references=mask['reference_ids'],
        evaluation_cases=400, expert_audit_cases=80, ppo_updates=0, stop_before_ppo=True,
        initial_model='new network', source_bowl_protocol='expert demonstrations + perturbations + three DAgger/BC rounds',
        geometry_validation=mask['validation'], allowed_references_min=min(len(c['allowed_reference_indices']) for c in cells),
        task_sha256=sha(OUT/'task.yaml')))
    print('CUP_PREPARED', mask['reference_ids'], checks, 'geometry checks', flush=True)


if __name__ == '__main__':
    main()
