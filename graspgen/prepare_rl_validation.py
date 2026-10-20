"""Build independent YCB physics assets/banks and conservative kitchen masks."""
import itertools
import json
import multiprocessing
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import trimesh
import yaml
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
from trimesh.collision import CollisionManager
from select_references import ROOT, OBJECTS, sha, origin
import xml.etree.ElementTree as ET

OUT = ROOT/'runs/ycb-rl-validation'
MASS_SOURCE = 'https://personalrobotics.cs.washington.edu/publications/calli2015ycb.pdf'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def make_usd(mesh, name, mass, path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, '/Object')
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    geom = UsdGeom.Mesh.Define(stage, '/Object/mesh')
    geom.CreatePointsAttr(mesh.vertices.tolist())
    geom.CreateFaceVertexCountsAttr([3]*len(mesh.faces))
    geom.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    geom.CreateSubdivisionSchemeAttr('none')
    geom.CreateDisplayColorAttr([Gf.Vec3f(.68,.31,.28) if name == 'mug' else Gf.Vec3f(.25,.42,.65)])
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr('convexDecomposition')
    geom.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
    geom.GetPrim().AddAppliedSchema('PhysxConvexDecompositionCollisionAPI')
    geom.GetPrim().CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.001)
    geom.GetPrim().CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
    # Fine decomposition is important for the mug cavity and both handles.
    geom.GetPrim().CreateAttribute('physxConvexDecompositionCollision:maxConvexHulls', Sdf.ValueTypeNames.Int).Set(64)
    geom.GetPrim().CreateAttribute('physxConvexDecompositionCollision:hullVertexLimit', Sdf.ValueTypeNames.Int).Set(64)
    geom.GetPrim().CreateAttribute('physxConvexDecompositionCollision:voxelResolution', Sdf.ValueTypeNames.Int).Set(500000)
    material = UsdShade.Material.Define(stage, '/Object/physicsMaterial')
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.)
    physics.CreateDynamicFrictionAttr(1.)
    physics.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material, materialPurpose='physics')
    stage.GetRootLayer().Save()


def hand_parts(opening_m=.08):
    assert 0 < opening_m <= .08
    cooked = pickle.load((ROOT/'assets/kitchen/cooked-model.pkl').open('rb'))
    link7_T_hand = np.array(json.loads((ROOT/'rl/assets/manifest.json').read_text())['link7_T_hand'])
    urdf = ET.parse(ROOT/'rl/assets/garmi_training.urdf').getroot()
    parts = []
    for record, mesh in cooked:
        body = record.get('body')
        if body == 'right_fr3_link7':
            transform = np.linalg.inv(link7_T_hand)
        elif body in ('right_fr3_leftfinger','right_fr3_rightfinger'):
            joint = next(j for j in urdf.findall('joint') if j.find('child').get('link') == body)
            transform = origin(joint.find('origin'))
            shift = np.eye(4)
            shift[:3,3] = np.fromstring(joint.find('axis').get('xyz'), sep=' ')*(opening_m/2)
            transform = transform@shift
        else:
            continue
        mesh = mesh.copy()
        mesh.apply_transform(transform)
        parts.append(mesh)
    assert len(parts) == 10
    return parts


_MASK_WORKER_STATE = None


def _mask_cell(cell):
    obstacles, sweeps, bounds, surfaces, height = _MASK_WORKER_STATE
    lo, hi = np.array(cell['bounds_map_xy'])
    xy = (lo+hi)/2
    surface = np.array(surfaces[0 if cell['region']=='left' else 1])
    footprint_ok = ((lo+bounds[0,:2] >= surface[0,:2]+.01)&(hi+bounds[1,:2] <= surface[1,:2]-.01)).all()
    translation = np.eye(4)
    translation[:3,3] = [xy[1]-6.35, .3-xy[0], height]
    allowed = []
    if footprint_ok:
        for j, sweep in enumerate(sweeps):
            if not any(obstacles.in_collision_single(part, transform=translation) for part in sweep):
                allowed.append(j)
    return dict(cell, allowed_reference_indices=allowed)


def build_mask(name, mesh, refs, bank_path):
    old = json.loads((ROOT/'assets/reference_filter/position-mask.json').read_text())
    obstacles = CollisionManager()
    for n, (record, obstacle) in enumerate(pickle.load((ROOT/'assets/kitchen/scene_meshes.pkl').open('rb'))):
        # Match the original mask's conservative mesh envelopes.
        obstacles.add_object(str(n), obstacle.convex_hull if record['type']=='Mesh' else obstacle)
    parts = hand_parts()
    rz = np.eye(4); rz[:3,:3] = [[0,1,0],[-1,0,0],[0,0,1]]
    height = .945-mesh.bounds[0,2]+.001
    shifts = np.array(list(itertools.product(*[[-.012,.012],[-.012,.012],[-.002,.002]])))
    sweeps = []
    for ref in refs:
        from scipy.spatial.transform import Rotation
        t = np.eye(4); t[:3,3] = ref['position']
        t[:3,:3] = Rotation.from_quat(np.array(ref['quaternion'])[[1,2,3,0]]).as_matrix()
        t = rz @ t
        sweep = []
        for part in parts:
            vertices = trimesh.transform_points(part.vertices, t)
            vertices = np.vstack([vertices, vertices-.08*t[:3,2]])
            hull = trimesh.convex.convex_hull((vertices[:,None,:]+shifts).reshape(-1,3))
            sweep.append(hull)
        sweeps.append(sweep)
    cells = []
    clear_counts = np.zeros(len(refs), int)
    # Check supported object footprint at all cell corners, including a 1 cm edge margin.
    bounds = mesh.bounds
    surfaces = [p['surface_bounds'] for p in json.loads((ROOT/'assets/reference_filter/proposal.json').read_text())['bowl_regions']]
    global _MASK_WORKER_STATE
    _MASK_WORKER_STATE = obstacles, sweeps, bounds, surfaces, height
    workers = int(os.environ.get('GARMI_MASK_WORKERS', '1'))
    assert workers >= 1
    # CPU-only geometry preparation; fork shares immutable FCL geometry. Never
    # invoke this pool from a process that has initialized a CUDA simulator.
    pool = multiprocessing.get_context('fork').Pool(workers) if workers > 1 else None
    try:
        mapped = pool.imap(_mask_cell, old['cells'], chunksize=8) if pool else map(_mask_cell, old['cells'])
        for ci, cell in enumerate(mapped):
            clear_counts[cell['allowed_reference_indices']] += 1
            cells.append(cell)
            if ci % 200 == 0: print('MASK', name, ci, flush=True)
        if pool:
            # Guard the forked execution against state or serialization drift.
            for ci in (0, len(cells)//2, len(cells)-1):
                assert cells[ci] == _mask_cell(old['cells'][ci])
    finally:
        if pool:
            pool.close(); pool.join()
        _MASK_WORKER_STATE = None
    # Empty cells are a real failure, not silently removed from the sampling region.
    empty = [i for i,c in enumerate(cells) if not c['allowed_reference_indices']]
    sources = {str(bank_path.relative_to(ROOT)):sha(bank_path)}
    for relative in ('assets/kitchen/scene_meshes.pkl','assets/kitchen/cooked-model.pkl',
                     'assets/kitchen/kitchen.usda','rl/assets/manifest.json','rl/assets/garmi_training.urdf'):
        sources[relative] = sha(ROOT/relative)
    mask = dict(schema_version=1, status='validated' if not empty else 'failed',
        object_name=name, bank_path=str(bank_path.relative_to(ROOT)), reference_ids=[r['id'] for r in refs],
        bowl_map_yaw_degrees=0., bowl_origin_z_m=float(height), cell_size_m=.02,
        numeric_position_tolerance_m=1.e-6, sources_sha256=sources,
        regions=old['regions'], cells=cells, extra_margin_m=.002,
        method='Cooked wrist/palm/open-finger convex solids, continuously swept across 2 cm cell and 8 cm straight pregrasp, plus 2 mm box margin. Mesh furniture uses convex envelopes.',
        scope='Static furniture clearance of hand/wrist approach only; not whole-arm reachability or grasp success.',
        empty_cells=empty, allowed_cells_per_reference=clear_counts.tolist())
    return mask


def crop_to_supported_footprint(mask, mesh):
    """Shrink the placement rectangle explicitly; never silently sample empty cells."""
    proposal=json.loads((ROOT/'assets/reference_filter/proposal.json').read_text())
    regions=[];cells=[]
    for region,surface_info in zip(mask['regions'],proposal['bowl_regions']):
        surface=np.array(surface_info['surface_bounds'])
        original=np.array(region['bounds_map_xy']);bounds=original.copy()
        bounds[0]=np.maximum(bounds[0],surface[0,:2]+.01-mesh.bounds[0,:2])
        bounds[1]=np.minimum(bounds[1],surface[1,:2]-.01-mesh.bounds[1,:2])
        bounds[0]=original[0]+np.ceil((bounds[0]-original[0])/mask['cell_size_m']-1.e-7)*mask['cell_size_m']
        bounds[1]=original[0]+np.floor((bounds[1]-original[0])/mask['cell_size_m']+1.e-7)*mask['cell_size_m']
        kept=[c for c in mask['cells'] if c['region']==region['name']
              and (np.array(c['bounds_map_xy'])[0]>=bounds[0]-1.e-7).all()
              and (np.array(c['bounds_map_xy'])[1]<=bounds[1]+1.e-7).all()]
        shape=np.rint((bounds[1]-bounds[0])/mask['cell_size_m']).astype(int)
        assert len(kept)==int(np.prod(shape)) and len(kept)>0
        regions.append(dict(region,bounds_map_xy=bounds.tolist(),shape_xy=shape.tolist(),cell_offset=len(cells)))
        cells.extend(kept)
    mask['regions']=regions;mask['cells']=cells
    mask['empty_cells']=[i for i,c in enumerate(cells) if not c['allowed_reference_indices']]
    mask['status']='failed' if mask['empty_cells'] else 'validated'
    mask['allowed_cells_per_reference']=[sum(i in c['allowed_reference_indices'] for c in cells) for i in range(len(mask['reference_ids']))]
    mask['placement_adjustment']='Original bowl rectangles clipped inward on 2 cm grid to keep the whole new object footprint at least 1 cm inside each worktop.'
    return mask


def main():
    for ycb, name in OBJECTS.items():
        out = OUT/ycb
        out.mkdir(parents=True, exist_ok=True)
        if name == 'pitcher':
            for file in ('selected-40.json','selected-40.npz'):
                shutil.copy2(ROOT/'runs/graspgen-ycb-40refs'/ycb/file, out/file)
        refs = json.loads((out/'selected-40.json').read_text())
        mesh_path = ROOT/'assets/ycb'/ycb/'google_16k/textured.obj'
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        mass = .118 if name == 'mug' else .178
        usd_path = ROOT/'assets/ycb'/ycb/'rl.usda'
        if not usd_path.exists(): make_usd(mesh, name, mass, usd_path)
        inside = lambda path: '/workspace/'+str(path.relative_to(ROOT))
        bank_path = out/'bank.json'
        bank = dict(status='diagnostic_validation_not_training_ready', objects=[dict(name=name,
            mesh=inside(mesh_path), mesh_sha256=sha(mesh_path), usd=inside(usd_path), usd_sha256=sha(usd_path),
            mass=mass, mass_source=MASS_SOURCE, friction_assumption=1., bounds=mesh.bounds.tolist(), references=refs)])
        save(bank_path, bank)
        if (out/'position-mask.json').exists():
            mask=json.loads((out/'position-mask.json').read_text())
            assert all(sha(ROOT/rel)==expected for rel,expected in mask['sources_sha256'].items())
        else:
            mask = build_mask(name, mesh, refs, bank_path)
        mask = crop_to_supported_footprint(mask,mesh)
        save(out/'position-mask.json', mask)
        cfg = yaml.safe_load((ROOT/'configs/task.yaml').read_text())
        task = cfg['garmi_task']
        task['multi_object'] = dict(manifest=inside(bank_path), objects=[name], split='all',
                                    reference_ids={name:[r['id'] for r in refs]})
        task['kitchen']['mask'] = inside(out/'position-mask.json')
        for region,valid in zip(task['kitchen']['bowl_regions'],mask['regions']):
            region['bounds_map_xy']=valid['bounds_map_xy']
        task['kitchen']['files_sha256'] = dict(mask=sha(out/'position-mask.json'), scene_usd=sha(ROOT/'assets/kitchen/kitchen.usda'))
        (out/'task.yaml').write_text(yaml.safe_dump(cfg))
        print('PREPARED', name, 'empty_cells', len(mask['empty_cells']),
              'reference_coverage', mask['allowed_cells_per_reference'], flush=True)


if __name__ == '__main__':
    main()
