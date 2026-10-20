"""Inspect PhysX's cooked hulls on the actual task stage, without asset edits."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--object',choices=('milk',),required=True)
parser.add_argument('--out',required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args).app
import json,sys
import numpy as np
import trimesh,yaml
from pxr import UsdUtils,PhysicsSchemaTools
import omni.physx,omni.usd
from isaaclab.envs import ManagerBasedRLEnv
from garmi_rl.config import configured_env
ROOT=Path(__file__).resolve().parents[1]
profile=yaml.safe_load((ROOT/'assets/milk/cook-task.yaml').read_text())
cfg,_=configured_env(profile,1,98765)
env=ManagerBasedRLEnv(cfg);env.forbidden_sensors=cfg.forbidden_sensors;env.reset()
stage=omni.usd.get_context().get_stage()
stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
if not stage_id:stage_id=UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
path='/World/envs/env_0/Object/mesh'
response={}
def received(result,convexes):
    response.update(result=str(result),convexes=convexes)
omni.physx.get_physx_cooking_interface().request_convex_collision_representation(
    stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
    run_asynchronously=False,on_result=received)
assert response.get('convexes'),response
meshes=[];hulls=[]
for convex in response['convexes']:
    vertices=np.array([[v.x,v.y,v.z] for v in convex.vertices])
    triangles=[]
    for polygon in convex.polygons:
        indices=convex.indices[polygon.index_base:polygon.index_base+polygon.num_vertices]
        triangles.extend([[indices[0],indices[j],indices[j+1]] for j in range(1,len(indices)-1)])
    mesh=trimesh.Trimesh(vertices=vertices,faces=np.asarray(triangles),process=False)
    meshes.append(mesh);hulls.append(dict(vertices=vertices.tolist(),triangles=triangles))
obj=env.grasp_bank.objects[0]
ref=env.grasp_bank.references[0][int(env.grasp_bank.reference_ids[0])]
combined=trimesh.util.concatenate(meshes)
clearance=None
limits=np.asarray(obj['bounds']);origin=np.array([[0.,0.,limits[1,2]+.10]])
hits,_,_=combined.ray.intersects_location(origin,np.array([[0.,0.,-1.]]),multiple_hits=True)
report=dict(object=args.object,result=response['result'],stage_collision_path=path,
    convex_parts=len(meshes),cooked_bounds=combined.bounds.tolist(),
    visual_bounds=limits.tolist(),cooked_vs_visual_bounds_max_error_m=float(np.abs(combined.bounds-limits).max()),
    cooked_centreline_surface_z_m=sorted(set(round(x,7) for x in hits[:,2]),reverse=True),
    open_gripper_clearance=clearance,actual_mass=env.scene['object'].root_physx_view.get_masses().tolist(),
    actual_inertia=env.scene['object'].root_physx_view.get_inertias().tolist())
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
(out/f'{args.object}-cooked.json').write_text(json.dumps(report,indent=2))
(out/f'{args.object}-hulls.json').write_text(json.dumps(hulls))
combined.export(out/f'{args.object}-cooked.ply')
print('COOKED',json.dumps(report),flush=True)
env.close();app.close()
