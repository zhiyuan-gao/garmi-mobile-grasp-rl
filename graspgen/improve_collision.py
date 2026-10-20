"""One bounded collision-proxy attempt; original USDs and references are untouched."""
import json
import pickle
from pathlib import Path
import numpy as np
import trimesh
from scipy.optimize import least_squares
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

ROOT=Path(__file__).resolve().parents[1]


def cut(mesh, normal, point):
    return trimesh.intersections.slice_mesh_plane(mesh,normal,point,cap=False)


def hull(mesh):
    if len(mesh.vertices)<4:return None
    h=trimesh.convex.convex_hull(mesh.vertices)
    return h if h.volume>1.e-11 else None


def fit_circle(mesh,z):
    center=np.array([-.02,.017])
    segments=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z])[:,:,:2]
    p=segments[:,0]-center;v=segments[:,1]-segments[:,0]
    points=[];thickness=[]
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for angle in np.linspace(.8,2*np.pi-.8,150):
        d=np.array([np.cos(angle),np.sin(angle)])
        den=cross(d,v);good=np.abs(den)>1.e-12
        t=cross(p,v)/np.where(good,den,1);u=cross(p,d)/np.where(good,den,1)
        t=t[good&(t>0)&(u>=0)&(u<=1)]
        if len(t):points.append(center+d*t.max());thickness.append(t.max()-t.min())
    points=np.array(points)
    fit=least_squares(lambda c:np.linalg.norm(points-c[:2],axis=1)-c[2],[-.02,.017,.041],loss='soft_l1',f_scale=.0005)
    return dict(z_m=z,center_xy_m=fit.x[:2].tolist(),radius_m=float(fit.x[2]),
                outer_residual_p95_m=float(np.percentile(abs(fit.fun),95)),wall_median_m=float(np.median(thickness)))


def slab_handle(mesh,normal,point,step):
    handle=cut(mesh,normal,point)
    handle.merge_vertices(merge_tex=True,merge_norm=True)
    parts=[]
    edges=np.linspace(handle.bounds[0,2],handle.bounds[1,2],int(np.ceil(np.ptp(handle.bounds[:,2])/step))+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        slab=cut(cut(handle,[0,0,1],[0,0,lo-1.e-5]),[0,0,-1],[0,0,hi+1.e-5])
        # A scanned tube may have disconnected inner/outer surfaces. Retain both;
        # selecting a single component can remove most of a handle cross-section.
        h=hull(slab)
        if h is not None:parts.append(h)
    return parts


def mug_parts(mesh):
    levels=[fit_circle(mesh,z) for z in [.003,.015,.04,.055,.073]]
    assert max(level['outer_residual_p95_m'] for level in levels)<.0006
    # Upper-wall scan estimates are ~1.2 mm; use 1.5 mm for the diagnostic proxy.
    wall=.0015
    parts=[]
    angles=np.linspace(0,2*np.pi,33)
    for lower,upper in zip(levels[:-1],levels[1:]):
        for a,b in zip(angles[:-1],angles[1:]):
            vertices=[]
            for level in (lower,upper):
                for angle in (a,b):
                    for radius in (level['radius_m'],level['radius_m']-wall):
                        xy=np.array(level['center_xy_m'])+radius*np.array([np.cos(angle),np.sin(angle)])
                        vertices.append([*xy,level['z_m']])
            parts.append(trimesh.convex.convex_hull(np.array(vertices)))
    # Bottom and upper rim retain the original scan, independently of fitted walls.
    parts.append(hull(cut(mesh,[0,0,-1],[0,0,.003])))
    rim=cut(mesh,[0,0,1],[0,0,.073-1.e-5])
    center=np.array([*levels[-1]['center_xy_m'],0.])
    for a,b in zip(angles[:-1],angles[1:]):
        n1=np.array([-np.sin(a),np.cos(a),0.]);n2=np.array([np.sin(b),-np.cos(b),0.])
        h=hull(cut(cut(rim,n1,center-1.e-5*n1),n2,center-1.e-5*n2))
        if h is not None:parts.append(h)
    body_count=len(parts)
    parts+=slab_handle(mesh,[1,0,0],[.021,0,0],.005)
    return parts,dict(method='Measured circular outer-wall profile, 32 angular sectors, 1.5 mm fitted wall; original scanned rim/bottom; handle divided into 5 mm height slabs.',
        wall_thickness_m=wall,profile=levels,body_piece_count=body_count,handle_piece_count=len(parts)-body_count,
        assumptions=['Lower inner-wall bulges removed by fitted thin-wall approximation; not an exact measured inner surface.',
                     'Original visual mesh unchanged; interior visual scan defects remain visible.',
                     'Mass/friction unchanged; inertia/COM estimated by PhysX from new collision solids.'])


def pitcher_parts(mesh,old):
    direction=np.array([.071,.069,0.]);direction/=np.linalg.norm(direction)
    point=np.array([-.015,.03,0.])+direction*.068
    parts=[]
    for part in old:
        h=hull(cut(part,-direction,point))
        if h is not None:parts.append(h)
    body_count=len(parts)
    parts+=slab_handle(mesh,direction,point,.008)
    return parts,dict(method='Keep previous body convexes inside handle split plane; replace outer handle with scanned-surface hulls in 8 mm height slabs.',
        split_plane_normal=direction.tolist(),split_plane_point=point.tolist(),body_piece_count=body_count,
        handle_piece_count=len(parts)-body_count,assumptions=['Body remains solid. No interior pitcher reconstruction.'])


def write_asset(source,target,parts):
    # Copy the old visual mesh and mass/material settings; replace only collision geometry.
    stage=Usd.Stage.Open(str(source));stage.GetRootLayer().Export(str(target))
    stage=Usd.Stage.Open(str(target))
    visual=stage.GetPrimAtPath('/Object/mesh')
    for schema in ['PhysicsCollisionAPI','PhysicsMeshCollisionAPI','PhysxCollisionAPI','PhysxConvexDecompositionCollisionAPI']:
        visual.RemoveAppliedSchema(schema)
    for prop in list(visual.GetProperties()):
        if prop.GetName().startswith(('physics:','physx')):visual.RemoveProperty(prop.GetName())
    material=UsdShade.Material.Get(stage,'/Object/physicsMaterial')
    for i,part in enumerate(parts):
        geo=UsdGeom.Mesh.Define(stage,f'/Object/collisions/part_{i:03d}')
        geo.CreatePointsAttr(part.vertices.tolist());geo.CreateFaceVertexCountsAttr([3]*len(part.faces))
        geo.CreateFaceVertexIndicesAttr(part.faces.reshape(-1).tolist());geo.CreateSubdivisionSchemeAttr('none')
        geo.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geo.GetPrim()).CreateApproximationAttr('convexHull')
        geo.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
        geo.GetPrim().AddAppliedSchema('PhysxConvexHullCollisionAPI')
        for name,value in [('physxCollision:contactOffset',.001),('physxCollision:restOffset',0.),('physxConvexHullCollision:minThickness',.0002)]:
            geo.GetPrim().CreateAttribute(name,Sdf.ValueTypeNames.Float).Set(value)
        UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(material,materialPurpose='physics')
    stage.GetRootLayer().Save()


def main():
    for y,name in [('025_mug','mug'),('019_pitcher_base','pitcher')]:
        asset=ROOT/'assets/ycb'/y;out=asset/'collision-v2';out.mkdir(exist_ok=False)
        mesh=trimesh.load(asset/'google_16k/textured.obj',force='mesh',process=False)
        old=[m for _,m in pickle.load((asset/'cooked.pkl').open('rb'))]
        parts,report=mug_parts(mesh) if name=='mug' else pitcher_parts(mesh,old)
        assert all(p is not None and p.is_watertight and p.volume>0 for p in parts)
        write_asset(asset/'rl.usda',out/'rl.usda',parts)
        with (out/'authored.pkl').open('wb') as f:pickle.dump(parts,f)
        report.update(status='candidate_not_validated',piece_count=len(parts))
        (out/'construction.json').write_text(json.dumps(report,indent=2)+'\n')
        print('COLLISION_CANDIDATE',y,len(parts),report,flush=True)


if __name__=='__main__':main()
