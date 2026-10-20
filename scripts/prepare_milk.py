"""Freeze the four historical milk grasps in the left-counter experiment."""
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import random
import numpy as np
import trimesh
import yaml
from cup_goal_geometry import GoalGeometry

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'assets/milk'
OUT=ROOT/'runs/milk4-left-expert-ppo'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    obj=json.loads((ASSETS/'bank.json').read_text())['objects'][0]
    refs=[r['id'] for r in obj['references']]
    assert refs==['milk-r004','milk-r008','milk-r018','milk-r036']
    config=yaml.safe_load((ROOT/'configs/task.yaml').read_text())
    settings=config['garmi_task']['kitchen']
    regions=[r for r in settings.pop('bowl_regions') if r['name']=='left']
    assert len(regions)==1
    regions[0]['probability']=1.
    settings.update(object_name='milk',object_regions=regions,
                    object_map_yaw_degrees=settings.pop('bowl_map_yaw_degrees'))
    geometry=GoalGeometry(ASSETS/'bank.json')
    queries=[geometry.query(i,(.01,.01),.002) for i in range(4)]
    exact=[geometry.query(i) for i in range(4)]
    lo,hi=np.asarray(regions[0]['bounds_map_xy']); shape=np.rint((hi-lo)/.02).astype(int)
    cells=[]; checks=0
    rng=np.random.default_rng(2026102101)
    for ix,iy in itertools.product(range(shape[0]),range(shape[1])):
        lower=lo+np.array([ix,iy])*.02;upper=lower+.02
        allowed=[j for j in range(4) if not queries[j].collides((lower+upper)/2)]
        if not allowed: raise ValueError(f'No allowed reference in left cell {ix},{iy}')
        points=[[lower[0],lower[1]],[lower[0],upper[1]],[upper[0],lower[1]],[upper[0],upper[1]],rng.uniform(lower,upper).tolist(),rng.uniform(lower,upper).tolist()]
        for j in allowed:
            for point in points:
                assert not exact[j].collides(point),(ix,iy,j,point)
                checks+=1
        cells.append(dict(region='left',index_xy=[ix,iy],bounds_map_xy=[lower.tolist(),upper.tolist()],allowed_reference_indices=allowed))
    original=pickle.load((ROOT/'assets/kitchen/cooked-model.pkl').open('rb'))
    cooked=[(r,m) for r,m in original if r['parent']!='Object']
    hulls=json.loads((ASSETS/'milk-hulls.json').read_text())
    for i,h in enumerate(hulls):
        cooked.append((dict(parent='Object',body='Object',path=f'milk/convex_{i}'),trimesh.Trimesh(h['vertices'],h['triangles'],process=False)))
    with (ASSETS/'cooked-model.pkl').open('wb') as f: pickle.dump(cooked,f)
    paths=['assets/milk/bank.json','assets/milk/milk.stl','assets/milk/milk.usda','assets/milk/milk-hulls.json','assets/milk/cooked-model.pkl','assets/kitchen/scene_meshes.pkl','assets/kitchen/kitchen.usda','rl/assets/manifest.json','rl/assets/garmi_training.urdf']
    hashes={p:sha(ROOT/p) for p in paths}
    mask=dict(schema_version=1,status='validated',object_name='milk',object_map_yaw_degrees=0.,
        object_origin_z_m=.946-obj['bounds'][0][2],reference_ids=refs,bank_path='assets/milk/bank.json',
        cell_size_m=.02,extra_margin_m=.002,numeric_position_tolerance_m=1.e-6,sources_sha256=hashes,
        regions=[dict(name='left',bounds_map_xy=regions[0]['bounds_map_xy'],shape_xy=shape.tolist(),cell_offset=0)],cells=cells,
        method='Swept convex hand geometry over 2 cm cells plus 2 mm margin',
        scope='Target hand vs closed furniture, not arm reachability or approach success',
        validation=dict(exact_corner_and_interior_checks=checks,violations=0))
    write(ASSETS/'position-mask.json',mask)
    write(ASSETS/'expert.json',dict(object_name='milk',bank='assets/milk/bank.json',cooked_model='assets/milk/cooked-model.pkl',files_sha256=hashes))
    settings.update(mask='/workspace/assets/milk/position-mask.json',files_sha256=dict(mask=sha(ASSETS/'position-mask.json'),scene_usd=sha(ROOT/'assets/kitchen/kitchen.usda')))
    config.pop('expert_bc',None);config.pop('evaluation',None)
    config['garmi_task']['multi_object']=dict(manifest='/workspace/assets/milk/bank.json',objects=['milk'],reference_ids={'milk':refs},split='all')
    OUT.mkdir(parents=True)
    forbidden=set()
    def sample(n,seed,prefix):
        rng=random.Random(seed); cases=[]
        for j,ref in enumerate(refs):
            pool=[c for c in cells if j in c['allowed_reference_indices']]
            assert pool,ref
            for k in range(n):
                lower,upper=rng.choice(pool)['bounds_map_xy']
                x,y=[rng.uniform(lower[a]+1.e-5,upper[a]-1.e-5) for a in range(2)]
                blo,bhi=settings['base_bounds_map_xy']; bx,by=[rng.uniform(blo[a],bhi[a]) for a in range(2)]
                case=dict(id=f'{prefix}-{ref}-{k:03}',object='milk',reference=ref,region='left',
                    depth='front' if y<7.363333 else 'rear' if y>7.496667 else 'middle',
                    robot_state=[by-6.35,.3-bx,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                    object_state=[y-6.35,.3-x,mask['object_origin_z_m'],2**-.5,0.,0.,-2**-.5,0.,0.,0.,0.,0.,0.])
                key=tuple(case['robot_state']+case['object_state']);assert key not in forbidden; forbidden.add(key);cases.append(case)
        rng.shuffle(cases);return cases
    write(OUT/'evaluation-cases.json',sample(100,2026102102,'evaluation'))
    write(OUT/'expert-audit-cases.json',sample(20,2026102103,'expert-audit'))
    config['evaluation']=dict(cases='/workspace/runs/milk4-left-expert-ppo/evaluation-cases.json',cases_sha256=sha(OUT/'evaluation-cases.json'))
    config['agent']['experiment'].update(directory='/workspace/runs/milk4-left-expert-ppo',experiment_name='milk_expert_ppo')
    (OUT/'task.yaml').write_text(yaml.safe_dump(config))
    write(OUT/'preparation.json',dict(status='prepared',object='milk',references=refs,evaluation_cases=400,
        expert_audit_cases=80,ppo_updates=1000,ppo_num_envs=4096,stop_before_ppo=False,initial_model='new network',
        placement_regions=['left'],task_sha256=sha(OUT/'task.yaml'),geometry_validation=mask['validation']))
    print('MILK_PREPARED',refs,checks,'geometry checks',flush=True)

if __name__=='__main__': main()
