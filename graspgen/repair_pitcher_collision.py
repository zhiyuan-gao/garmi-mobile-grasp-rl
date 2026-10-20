"""Fit the existing pitcher scan with shallow convex slabs; preserve visual and reference data."""
import argparse,json,pickle
from pathlib import Path
import numpy as np,trimesh,manifold3d as md
from scipy.spatial import ConvexHull
from improve_collision import cut,write_asset
from select_references import ROOT,sha

ASSET=ROOT/'assets/ycb/019_pitcher_base'
OUT=ASSET/'contact-v3'


def stable_hull(vertices):
    # Quantize far below the 1 mm contact offset, avoiding nearly coincident
    # slice vertices that can make generic hull cleanup drop necessary faces.
    v=np.unique(np.round(vertices,7),axis=0)
    if len(v)<4:return None
    h=ConvexHull(v)
    if h.volume<1e-11:return None
    faces=h.simplices.copy()
    normals=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])
    flip=(normals*h.equations[:,:3]).sum(1)<0;faces[flip]=faces[flip][:,[0,2,1]]
    p=trimesh.Trimesh(v,faces,process=False);p.remove_unreferenced_vertices()
    assert p.is_watertight and p.is_winding_consistent and p.volume>0
    return p


def slabs(mesh,step):
    edges=np.linspace(mesh.bounds[0,2],mesh.bounds[1,2],int(np.ceil(np.ptp(mesh.bounds[:,2])/step))+1)
    result=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        clipped=cut(cut(mesh,[0,0,1],[0,0,lo-1e-5]),[0,0,-1],[0,0,hi+1e-5])
        p=stable_hull(clipped.vertices)
        if p is not None:
            p=stable_hull(p.centroid+(p.vertices-p.centroid)*1.0002)
            result.append(p)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--handle-step',type=float,default=.001)
    args=parser.parse_args();out=args.out.resolve()
    if args.handle_step<=0:parser.error('--handle-step must be positive')
    out.mkdir(exist_ok=False)
    source=ASSET/'google_16k/textured.obj'
    mesh=trimesh.load(source,force='mesh',process=False)
    d=np.array([.071,.069,0.]);d/=np.linalg.norm(d)
    split=np.array([-.015,.03,0.])+d*.068
    body=slabs(cut(mesh,-d,split),.004)
    handle=slabs(cut(mesh,d,split),args.handle_step)
    parts=body+handle
    solids=[md.Manifold(md.Mesh64(np.asarray(p.vertices),np.asarray(p.faces,dtype=np.uint64))) for p in parts]
    assert all(s.status()==md.Error.NoError for s in solids)
    union=md.Manifold.batch_boolean(solids,md.OpType.Add)
    assert union.status()==md.Error.NoError and len(union.decompose())==1 and union.genus()==1
    exported=union.to_mesh64();boundary=trimesh.Trimesh(exported.vert_properties[:,:3],exported.tri_verts,process=False)
    assert boundary.is_watertight and boundary.is_winding_consistent
    boundary.export(out/'authored-boundary.obj',digits=12)
    with (out/'authored.pkl').open('wb') as f:pickle.dump(parts,f)
    write_asset(ASSET/'rl.usda',out/'rl.usda',parts)
    report=dict(status='candidate_pending_cooked_contact_audit',source_mesh_sha256=sha(source),
        method='Solid body and handle fitted to scan in shallow height slabs; explicit closed convex hulls.',
        body_step_m=.004,handle_step_m=args.handle_step,
        body_pieces=len(body),handle_pieces=len(handle),total_pieces=len(parts),
        coordinate_quantization_m=1e-7,relative_overlap=.0002,slab_z_overlap_m=2e-5,
        split_plane_normal=d.tolist(),split_plane_point=split.tolist(),union_components=1,union_genus=1,
        visual_mesh_changed=False,reference_poses_changed=False,
        assumptions=['Body remains solid. This does not reconstruct or validate the pitcher interior.',
            'Mass and friction unchanged; PhysX recomputes inertia/COM for modified collision shapes.'])
    (out/'repair.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
