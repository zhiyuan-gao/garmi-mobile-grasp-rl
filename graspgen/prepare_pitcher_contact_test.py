"""Keep original pitcher references/cases; test the corrected collision asset separately."""
import argparse,json,pickle,itertools
from pathlib import Path
import numpy as np,trimesh,yaml
from trimesh.collision import CollisionManager
from pxr import Usd
from select_references import ROOT,sha

old=ROOT/'runs/ycb-rl-validation/019_pitcher_base'
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--out',type=Path,default=ROOT/'runs/pitcher-handle-audit/after')
parser.add_argument('--asset',type=Path,default=ROOT/'assets/ycb/019_pitcher_base/contact-v3')
args=parser.parse_args();out=args.out.resolve();asset=args.asset.resolve()
out.mkdir(parents=True,exist_ok=True)
bank=json.loads((old/'bank.json').read_text());assert not (out/'bank.json').exists()
bank['status']='corrected_pitcher_collision_diagnostic_expert_pending'
bank['objects'][0].update(usd='/workspace/'+str((asset/'rl.usda').relative_to(ROOT)),usd_sha256=sha(asset/'rl.usda'))
(out/'bank.json').write_text(json.dumps(bank,indent=2)+'\n')
mask=json.loads((old/'position-mask.json').read_text())
del mask['sources_sha256'][mask['bank_path']]
mask['bank_path']=str((out/'bank.json').relative_to(ROOT));mask['sources_sha256'][mask['bank_path']]=sha(out/'bank.json')
(out/'position-mask.json').write_text(json.dumps(mask,indent=2)+'\n')
cfg=yaml.safe_load((old/'task.yaml').read_text())
cfg['garmi_task']['multi_object']['manifest']='/workspace/'+str((out/'bank.json').relative_to(ROOT))
cfg['garmi_task']['kitchen']['mask']='/workspace/'+str((out/'position-mask.json').relative_to(ROOT))
cfg['garmi_task']['kitchen']['files_sha256']['mask']=sha(out/'position-mask.json')
(out/'task.yaml').write_text(yaml.safe_dump(cfg))
for path,digest in mask['sources_sha256'].items():assert sha(ROOT/path)==digest
meta=json.loads((asset/'cooked.json').read_text())
assert sha(asset/'rl.usda')==meta['usd_sha256'] and sha(asset/'cooked.pkl')==meta['cooked_sha256']
s0=Usd.Stage.Open(str(ROOT/'assets/ycb/019_pitcher_base/rl.usda'));s1=Usd.Stage.Open(str(asset/'rl.usda'))
assert all(s0.GetPrimAtPath('/Object/mesh').GetAttribute(k).Get()==s1.GetPrimAtPath('/Object/mesh').GetAttribute(k).Get() for k in ['points','faceVertexCounts','faceVertexIndices'])
parts=[p for _,p in pickle.load((asset/'cooked.pkl').open('rb'))];bounds=trimesh.util.concatenate(parts).bounds
scene=CollisionManager()
for i,(r,m) in enumerate(pickle.load((ROOT/'assets/kitchen/scene_meshes.pkl').open('rb'))):scene.add_object(str(i),m.convex_hull if r['type']=='Mesh' else m)
surfaces=[np.array(r['surface_bounds']) for r in json.loads((ROOT/'assets/reference_filter/proposal.json').read_text())['bowl_regions']]
margins=[]
for region,surface in zip(mask['regions'],surfaces):
    lo,hi=np.array(region['bounds_map_xy']);gaps=np.vstack([lo+bounds[0,:2]-surface[0,:2],surface[1,:2]-hi-bounds[1,:2]])
    margins.append(dict(region=region['name'],minimum_edge_gap_m=float(gaps.min())))
assert min(m['minimum_edge_gap_m'] for m in margins)>=.01
rz=np.array([[0,1,0],[-1,0,0],[0,0,1]]);shifts=np.array(list(itertools.product([-.01,.01],[-.01,.01],[0.])))
swept=[trimesh.convex.convex_hull(((p.vertices@rz.T)[:,None,:]+shifts).reshape(-1,3)) for p in parts]
manager=CollisionManager();manager.add_object('swept',trimesh.util.concatenate(swept));collisions=[]
for i,c in enumerate(mask['cells']):
    xy=np.mean(c['bounds_map_xy'],axis=0);t=np.eye(4);t[:3,3]=[xy[1]-6.35,.3-xy[0],mask['bowl_origin_z_m']]
    manager.set_transform('swept',t)
    if manager.in_collision_other(scene):collisions.append(i)
assert not collisions
audit=dict(status='passed',references_unchanged=True,visual_mesh_unchanged=True,sampling_mesh_unchanged=True,
    hash_checks_passed=True,placement_cells=len(mask['cells']),object_furniture_collision_cells=collisions,
    worktop_margins=margins,actual_cooked_bounds_m=bounds.tolist(),
    initial_bottom_clearance_m=float(mask['bowl_origin_z_m']+bounds[0,2]-.945),
    unchanged=['reference poses','placement cells','mass','friction','controller','seed','source surface for endpoint precheck'],
    limitations=['Only collision geometry changed; PhysX recomputes inertia and COM.',
        'Single finite diagnostic run; no expert approach trajectories or training.'])
(out/'input-and-placement-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2),flush=True)
