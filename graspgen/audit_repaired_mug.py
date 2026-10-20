"""Audit saved geometry, reference frames, hashes and whole-object kitchen placement."""
import json,pickle,itertools
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
from trimesh.collision import CollisionManager
from pxr import Usd,UsdGeom
from select_references import ROOT,HAND_OFFSET,sha
from prepare_repaired_mug import OUT,ASSET

bank=json.loads((OUT/'bank.json').read_text());obj=bank['objects'][0]
mask=json.loads((OUT/'position-mask.json').read_text())
for relative,expected in mask['sources_sha256'].items():assert sha(ROOT/relative)==expected
for key in ['mesh','usd']:assert sha(ROOT/obj[key].removeprefix('/workspace/'))==obj[key+'_sha256']
meta=json.loads((ASSET/'cooked.json').read_text())
assert sha(ASSET/'rl.usda')==meta['usd_sha256'] and sha(ASSET/'cooked.pkl')==meta['cooked_sha256']
refs=obj['references'];poses=np.load(OUT/'selected-40.npz')['T_object_graspgen']
assert len(refs)==40 and mask['reference_ids']==[r['id'] for r in refs]
for ref,pose in zip(refs,poses):
    t=pose@HAND_OFFSET;q=np.array(ref['quaternion'])
    assert np.allclose(t[:3,:3],Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),atol=1e-6)
    assert np.allclose(t[:3,3],ref['position'],atol=1e-8)
mesh=trimesh.load(ASSET/'repaired.obj',force='mesh',process=False)
assert mesh.is_watertight and mesh.euler_number==0
stage=Usd.Stage.Open(str(ASSET/'rl.usda'));visual=UsdGeom.Mesh.Get(stage,'/Object/mesh')
assert np.allclose(np.array(visual.GetPointsAttr().Get()),mesh.vertices,atol=1e-8)
parts=[p for _,p in pickle.load((ASSET/'cooked.pkl').open('rb'))]
bounds=trimesh.util.concatenate(parts).bounds
scene=CollisionManager()
for i,(r,m) in enumerate(pickle.load((ROOT/'assets/kitchen/scene_meshes.pkl').open('rb'))):scene.add_object(str(i),m.convex_hull if r['type']=='Mesh' else m)
surfaces=[np.array(r['surface_bounds']) for r in json.loads((ROOT/'assets/reference_filter/proposal.json').read_text())['bowl_regions']]
margins=[]
for region,surface in zip(mask['regions'],surfaces):
    lo,hi=np.array(region['bounds_map_xy']);gaps=np.vstack([lo+bounds[0,:2]-surface[0,:2],surface[1,:2]-hi-bounds[1,:2]])
    margins.append(dict(region=region['name'],minimum_worktop_edge_gap_m=float(gaps.min())))
assert min(m['minimum_worktop_edge_gap_m'] for m in margins)>=.01
rz=np.array([[0,1,0],[-1,0,0],[0,0,1]])
shifts=np.array(list(itertools.product([-.01,.01],[-.01,.01],[0.])))
swept=[trimesh.convex.convex_hull(((p.vertices@rz.T)[:,None,:]+shifts).reshape(-1,3)) for p in parts]
manager=CollisionManager();manager.add_object('swept',trimesh.util.concatenate(swept));collisions=[]
for i,c in enumerate(mask['cells']):
    xy=np.mean(c['bounds_map_xy'],axis=0);t=np.eye(4);t[:3,3]=[xy[1]-6.35,.3-xy[0],mask['bowl_origin_z_m']]
    manager.set_transform('swept',t)
    if manager.in_collision_other(scene):collisions.append(i)
assert not collisions
report=dict(status='passed',reference_frames_checked=40,hashes_verified=True,visual_and_sampling_vertices_match=True,
    sampling_mesh_watertight=True,sampling_mesh_genus=1,collision_pieces=len(parts),
    actual_cooked_bounds_m=bounds.tolist(),worktop_margins=margins,checked_placement_cells=len(mask['cells']),
    object_furniture_collision_cells=collisions,initial_bottom_clearance_m=float(mask['bowl_origin_z_m']+bounds[0,2]-.945),
    references_without_placement=[refs[i]['id'] for i,c in enumerate(mask['allowed_cells_per_reference']) if not c],
    expert_trajectories_generated=0,optimizer_updates=0)
(OUT.parent/'input-and-placement-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
