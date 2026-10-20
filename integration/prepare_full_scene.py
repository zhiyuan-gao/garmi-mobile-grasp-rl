"""Add the complete authored GARMI apartment around the frozen RL kitchen.

Furniture stays at its authored closed pose. The existing kitchen colliders are
retained exactly; other apartment colliders are copied with their world transform.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def main():
    root = Path('/workspace')
    out = root / 'integration/assets'
    out.mkdir(exist_ok=True)
    source_path = root / 'cram-vrb-lab/assets/garmi-apartment/world.usda'
    kitchen_path = root / 'results/rl-preflight/bowl-kitchen-full/kitchen.usda'
    prior = json.loads(kitchen_path.with_name('scene.json').read_text())
    kept = {item['path'] for item in prior['colliders']}
    source = Usd.Stage.Open(str(source_path))
    flat = source.Flatten()
    stage = Usd.Stage.CreateNew(str(out / 'apartment.usda'))
    apartment = UsdGeom.Xform.Define(stage, '/Apartment').GetPrim()
    apartment.GetReferences().AddReference(str(kitchen_path))
    stage.SetDefaultPrim(apartment)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    for item in prior['colliders']:
        path = item['output_path'].replace('/Kitchen', '/Apartment', 1)
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr('invisible')
    transform = np.array([[0., 1., 0., -6.35], [-1., 0., 0., .3],
                          [0., 0., 1., 0.], [0., 0., 0., 1.]])
    visual_parent = UsdGeom.Xform.Define(stage, '/Apartment/MapVisuals')
    visual_parent.AddTransformOp().Set(Gf.Matrix4d(transform.T.tolist()))
    Sdf.CopySpec(flat, '/Root', stage.GetRootLayer(), '/Apartment/MapVisuals/Scene')
    visuals = stage.GetPrimAtPath('/Apartment/MapVisuals/Scene')
    remove = []
    for prim in Usd.PrimRange(visuals):
        if prim.IsA(UsdPhysics.Joint):
            remove.append(prim.GetPath())
            continue
        for schema in list(prim.GetAppliedSchemas()):
            if schema.startswith(('Physics', 'Physx')):
                prim.RemoveAppliedSchema(schema)
    for path in reversed(remove):
        stage.RemovePrim(path)
    material = UsdShade.Material(stage.GetPrimAtPath('/Apartment/Material'))
    cache = UsdGeom.XformCache()
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
    added = []
    for prim in source.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or str(prim.GetPath()) in kept:
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.GetMax()[2] < .01:  # The RL simulator supplies the floor collider.
            continue
        target = f'/Apartment/room_collider_{len(added):03d}'
        Sdf.CopySpec(flat, prim.GetPath(), stage.GetRootLayer(), target)
        copied = stage.GetPrimAtPath(target)
        for child in list(copied.GetChildren()):
            stage.RemovePrim(child.GetPath())
        for schema in list(copied.GetAppliedSchemas()):
            if 'RigidBody' in schema or 'Mass' in schema or 'Articulation' in schema:
                copied.RemoveAppliedSchema(schema)
        for rel in list(copied.GetRelationships()):
            if rel.GetName().startswith(('material:binding', 'physics:')):
                copied.RemoveProperty(rel.GetName())
        xf = UsdGeom.Xformable(copied)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d((transform @ np.array(cache.GetLocalToWorldTransform(prim)).T).T.tolist()))
        UsdPhysics.CollisionAPI.Apply(copied).CreateCollisionEnabledAttr(True)
        copied.AddAppliedSchema('PhysxCollisionAPI')
        copied.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.001)
        copied.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
        UsdShade.MaterialBindingAPI.Apply(copied).Bind(material, materialPurpose='physics')
        UsdGeom.Imageable(copied).CreateVisibilityAttr('invisible')
        added.append(dict(source=str(prim.GetPath()), output=target,
                          map_bounds=[list(box.GetMin()), list(box.GetMax())]))
    stage.GetRootLayer().Save()
    report = dict(source=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  kitchen_sha256=hashlib.sha256(kitchen_path.read_bytes()).hexdigest(),
                  retained_kitchen_colliders=len(kept), added_colliders=added,
                  furniture_frozen_at_authored_pose=True, complete_apartment_visuals=True)
    (out / 'apartment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Kept {len(kept)} kitchen colliders; added {len(added)} apartment colliders.')


if __name__ == '__main__':
    main()
