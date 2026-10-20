"""Build one continuous sampling surface and matching collision pieces; no grasps."""
import json,pickle,hashlib
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as md
from pxr import Usd,UsdGeom
from improve_collision import write_asset

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'assets/ycb/025_mug'
OUT=ASSET/'surface-v3'
RUN=ROOT/'runs/mug-surface-v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def inside_parts(points,parts):
    from scipy.spatial import ConvexHull
    inside=np.zeros(len(points),dtype=bool)
    for p in parts:
        keep=np.flatnonzero(~inside & (points>=p.bounds[0]-1e-10).all(1) & (points<=p.bounds[1]+1e-10).all(1))
        if len(keep):
            eq=ConvexHull(p.vertices).equations
            inside[keep]=((points[keep]@eq[:,:3].T+eq[:,3])<=1e-10).all(1)
    return inside


def main():
    OUT.mkdir(exist_ok=False);RUN.mkdir(exist_ok=True)
    source=ASSET/'collision-v2/authored.pkl'
    parts=pickle.load(source.open('rb'))
    count=json.loads((ASSET/'collision-v2/construction.json').read_text())['body_piece_count']
    # Extend only cut-plane vertices at the handle roots into the fitted wall.
    for i in range(count,len(parts)):
        vertices=parts[i].vertices.copy()
        vertices[np.abs(vertices[:,0]-.021)<1e-7,0]=.018
        parts[i]=trimesh.convex.convex_hull(vertices)
    # Microscopic overlap removes numerical seams between exactly touching sectors.
    maximum_overlap_shift=0.
    for part in parts:
        shift=(part.vertices-part.centroid)*.002
        maximum_overlap_shift=max(maximum_overlap_shift,float(np.linalg.norm(shift,axis=1).max()))
        part.vertices=part.vertices+shift
    solids=[md.Manifold(md.Mesh64(np.asarray(p.vertices),np.asarray(p.faces,dtype=np.uint64))) for p in parts]
    assert all(s.status()==md.Error.NoError for s in solids)
    union=md.Manifold.batch_boolean(solids,md.OpType.Add).set_tolerance(1e-8)
    assert union.status()==md.Error.NoError and len(union.decompose())==1 and union.genus()==1
    exported=union.to_mesh64()
    mesh=trimesh.Trimesh(exported.vert_properties[:,:3],exported.tri_verts,process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0 and mesh.euler_number==0
    mesh_path=OUT/'repaired.obj';mesh.export(mesh_path,digits=12)
    # Check the saved mesh too, because it is the GraspGen input.
    loaded=trimesh.load(mesh_path,force='mesh',process=False)
    assert loaded.is_watertight and loaded.is_winding_consistent and loaded.euler_number==0
    points,faces=trimesh.sample.sample_surface(loaded,20000,seed=20261018)
    normals=loaded.face_normals[faces]
    inward=inside_parts(points-normals*2e-6,parts)
    outward=inside_parts(points+normals*2e-6,parts)
    boundary_ok=inward & ~outward
    assert boundary_ok.mean()>=.999, f'Boundary probe pass rate {boundary_ok.mean()}'
    # These points must stay free: cup cavity and the central handle aperture.
    probes=np.array([[-.0207,.0174,z] for z in [.01,.02,.04,.06,.075]]+[[.030,.015,.040]])
    assert not inside_parts(probes,parts).any()
    with (OUT/'authored.pkl').open('wb') as f:pickle.dump(parts,f)
    write_asset(ASSET/'rl.usda',OUT/'rl.usda',parts)
    stage=Usd.Stage.Open(str(OUT/'rl.usda'));geo=UsdGeom.Mesh.Get(stage,'/Object/mesh')
    geo.GetPointsAttr().Set(loaded.vertices.tolist());geo.GetFaceVertexCountsAttr().Set([3]*len(loaded.faces))
    geo.GetFaceVertexIndicesAttr().Set(loaded.faces.reshape(-1).tolist());stage.GetRootLayer().Save()
    np.savez_compressed(RUN/'sampling-preview.npz',points=points,face_indices=faces)
    report=dict(status='continuous_surface_verified_pending_grasp_generation',
        source_authored_sha256=sha(source),source_original_mesh_sha256=sha(ASSET/'google_16k/textured.obj'),
        sampling_mesh_sha256=sha(mesh_path),usd_sha256=sha(OUT/'rl.usda'),
        method='Float64 boolean union boundary of fitted body/rim/bottom and scan-based handle solids; no sampling of internal convex faces.',
        handle_root_extension_m=.003,overlap_scale=1.002,maximum_overlap_vertex_shift_m=maximum_overlap_shift,
        boolean_simplification_tolerance_m=1e-8,watertight=bool(loaded.is_watertight),
        connected_solids=1,genus=1,euler_number=int(loaded.euler_number),vertices=len(loaded.vertices),faces=len(loaded.faces),
        bounds_m=loaded.bounds.tolist(),volume_m3=float(loaded.volume),collision_pieces=len(parts),
        boundary_probe_count=len(points),boundary_probe_offset_m=2e-6,boundary_probe_pass_fraction=float(boundary_ok.mean()),
        free_cavity_handle_probes_m=probes.tolist(),assumptions=[
            'Wall thickness 1.5 mm remains an approximation, not a measured true inner surface.',
            'Handle root gap bridged automatically; geometry is a repaired proxy, not the unmodified YCB scan.',
            'Point cloud is sampled over the full surface, not a simulated camera view.',
            'Original YCB assets and previous reference banks are untouched.'])
    (OUT/'repair.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
