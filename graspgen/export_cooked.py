"""Export the actual PhysX object convexes for the later Giskard preflight."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(__doc__)
p.add_argument('--asset', action='append', help='Explicit USD asset; repeat for multiple candidates.')
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();app=AppLauncher(args).app
import hashlib
import json
import pickle
import numpy as np
import trimesh
import omni.usd
from omni.physx import get_physx_cooking_interface
from pxr import UsdUtils, PhysicsSchemaTools, UsdPhysics, UsdGeom

try:
    assets=[Path(a) for a in args.asset] if args.asset else [Path('/workspace/assets/ycb')/name/'rl.usda' for name in ('019_pitcher_base','025_mug')]
    for asset in assets:
        context=omni.usd.get_context()
        context.open_stage(str(asset))
        stage=context.get_stage()
        stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        pieces=[]
        def ready(result, convexes):
            for i,convex in enumerate(convexes):
                vertices=np.array([[v.x,v.y,v.z] for v in convex.vertices])
                triangles=[]
                for polygon in convex.polygons:
                    indices=convex.indices[polygon.index_base:polygon.index_base+polygon.num_vertices]
                    triangles.extend([[indices[0],indices[j],indices[j+1]] for j in range(1,len(indices)-1)])
                pieces.append((dict(parent='Object',body='Object',part=len(pieces)),
                    trimesh.Trimesh(vertices=vertices,faces=triangles,process=False)))
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                before=len(pieces)
                get_physx_cooking_interface().request_convex_collision_representation(
                    stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
                    run_asynchronously=False,on_result=ready)
                if len(pieces)==before:raise RuntimeError(f'PhysX returned no convexes for {asset}: {prim.GetPath()}')
        if not pieces:raise RuntimeError(f'PhysX returned no convexes for {asset}')
        out=asset.parent/'cooked.pkl'
        with out.open('wb') as f:pickle.dump(pieces,f)
        metadata=dict(usd_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
            cooked_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),convex_count=len(pieces))
        (asset.parent/'cooked.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print('EXPORTED_COOKED',str(asset),len(pieces),flush=True)
finally:
    app.close()
