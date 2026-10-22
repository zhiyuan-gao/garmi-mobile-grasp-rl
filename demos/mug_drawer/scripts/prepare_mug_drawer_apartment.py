"""Surround the physical drawer demo with the authored GARMI apartment.

Run through scripts/sim.sh after prepare_mug_drawer.py. Existing kitchen and
drawer physics are preserved. Authored visuals replace the visible colliders;
room colliders outside the existing kitchen are retained as static geometry.
"""
import os
import sys
from pathlib import Path

# USD can author these assets without starting a simulator/GPU context.
lib = '/isaac-sim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311'
if not os.environ.get('USD_READY'):
    os.environ['USD_READY'] = '1'
    os.environ['LD_LIBRARY_PATH'] = lib + '/bin:' + os.environ.get('LD_LIBRARY_PATH', '')
    os.execv(sys.executable, [sys.executable, *sys.argv])
sys.path.insert(0, lib)
import hashlib
import json
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

out = Path('/demo/assets/mug-drawer')
source_path = Path('/home/jovyan/cram-vrb-lab/assets/garmi-apartment/world.usda')
source = Usd.Stage.Open(str(source_path))
flat = source.Flatten()
prior = json.loads(Path('/workspace/assets/kitchen/scene.json').read_text())
kept = {item['path'] for item in prior['colliders']}
task_T_map = np.array([[0., 1., 0., -6.35], [-1., 0., 0., .3],
                       [0., 0., 1., 0.], [0., 0., 0., 1.]])
cache = UsdGeom.XformCache()
drawer_source = '/Root/Meshes/Assets/cabinet/Actor_0000/drawer_1'


def visual_only(stage, root):
    remove = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint):
            remove.append(prim.GetPath())
            continue
        for schema in list(prim.GetAppliedSchemas()):
            if schema.startswith(('Physics', 'Physx')):
                prim.RemoveAppliedSchema(schema)
    for path in reversed(remove):
        stage.RemovePrim(path)


# Replace the static kitchen's collision display with full authored visuals.
Usd.Stage.Open(str(out/'kitchen.usda')).Flatten().Export(str(out/'apartment.usda'))
stage = Usd.Stage.Open(str(out/'apartment.usda'))
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
vp = UsdGeom.Xform.Define(stage, '/Kitchen/MapVisuals')
vp.AddTransformOp().Set(Gf.Matrix4d(task_T_map.T.tolist()))
Sdf.CopySpec(flat, '/Root', stage.GetRootLayer(), '/Kitchen/MapVisuals/Scene')
visual_only(stage, stage.GetPrimAtPath('/Kitchen/MapVisuals/Scene'))
stage.RemovePrim(drawer_source.replace('/Root', '/Kitchen/MapVisuals/Scene', 1))

material = UsdShade.Material(stage.GetPrimAtPath('/Kitchen/Material'))
bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
added = []
for prim in source.Traverse():
    if not prim.HasAPI(UsdPhysics.CollisionAPI) or str(prim.GetPath()) in kept:
        continue
    if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
        continue
    if bounds.ComputeWorldBound(prim).ComputeAlignedRange().GetMax()[2] < .01:
        continue  # The existing simulator supplies the floor collider.
    target = f'/Kitchen/room_collider_{len(added):03d}'
    Sdf.CopySpec(flat, prim.GetPath(), stage.GetRootLayer(), target)
    copied = stage.GetPrimAtPath(target)
    for child in list(copied.GetChildren()):
        stage.RemovePrim(child.GetPath())
    for schema in list(copied.GetAppliedSchemas()):
        if any(key in schema for key in ['RigidBody', 'Mass', 'Articulation']):
            copied.RemoveAppliedSchema(schema)
    for rel in list(copied.GetRelationships()):
        if rel.GetName().startswith(('material:binding', 'physics:')):
            copied.RemoveProperty(rel.GetName())
    xf = UsdGeom.Xformable(copied)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d((task_T_map @ np.array(cache.GetLocalToWorldTransform(prim)).T).T.tolist()))
    UsdPhysics.CollisionAPI.Apply(copied).CreateCollisionEnabledAttr(True)
    copied.AddAppliedSchema('PhysxCollisionAPI')
    copied.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.001)
    copied.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
    UsdShade.MaterialBindingAPI.Apply(copied).Bind(material, materialPurpose='physics')
    UsdGeom.Imageable(copied).CreateVisibilityAttr('invisible')
    added.append(str(prim.GetPath()))
stage.GetRootLayer().Save()

# The original drawer appearance follows the same passive rigid body/rail.
Usd.Stage.Open(str(out/'drawer.usda')).Flatten().Export(str(out/'drawer_visual.usda'))
drawer = Usd.Stage.Open(str(out/'drawer_visual.usda'))
for prim in drawer.Traverse():
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
Sdf.CopySpec(flat, drawer_source, drawer.GetRootLayer(), '/Drawer/Visual')
visual = drawer.GetPrimAtPath('/Drawer/Visual')
visual_only(drawer, visual)
local = task_T_map @ np.array(cache.GetLocalToWorldTransform(source.GetPrimAtPath(drawer_source))).T
local[:3, 3] -= [1.053045, .39, .8]
xf = UsdGeom.Xformable(visual)
xf.ClearXformOpOrder()
xf.AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))
# Keep materials INSIDE the default prim so USD references retain bindings.
materials = {}
for prim in Usd.PrimRange(visual):
    for rel in prim.GetRelationships():
        if rel.GetName().startswith('material:binding'):
            targets = []
            for old in rel.GetTargets():
                if old not in materials:
                    new = Sdf.Path(f'/Drawer/Materials/material_{len(materials):02d}')
                    Sdf.CreatePrimInLayer(drawer.GetRootLayer(), new.GetParentPath())
                    Sdf.CopySpec(flat, old, drawer.GetRootLayer(), new)
                    materials[old] = new
                targets.append(materials[old])
            rel.SetTargets(targets)
drawer.GetRootLayer().Save()
report = dict(source=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
              full_apartment=True, added_room_colliders=added, drawer_visual_source=drawer_source,
              preserved_kitchen_and_drawer_physics=True)
(out/'apartment-manifest.json').write_text(json.dumps(report, indent=2))
print('APARTMENT_READY', len(added), 'extra room colliders', flush=True)
