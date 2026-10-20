"""Static PNG comparison of original and candidate actual PhysX convexes."""
from pathlib import Path
import json, pickle
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/ycb-collision-v2';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(3,2,figsize=(12,14))
report={}
for row,(y,title,origin,u,v) in enumerate([
    ('025_mug','Mug: handle and cavity',[0,.015,0],[1,0,0],[0,0,1]),
    ('025_mug','Mug: horizontal section at 20 mm',[0,0,.020],[1,0,0],[0,1,0]),
    ('019_pitcher_base','Pitcher: handle',[-.015,.03,0],[.071,.069,0],[0,0,1])]):
    asset=ROOT/'assets/ycb'/y
    mesh=trimesh.load(asset/'google_16k/textured.obj',force='mesh',process=False)
    u=np.array(u,dtype=float);u/=np.linalg.norm(u);v=np.array(v,dtype=float)
    normal=np.cross(u,v);o=np.array(origin);basis=np.stack([u,v],axis=1)
    for col,version in enumerate(['original','collision-v2']):
        folder=asset if col==0 else asset/version
        parts=[m for _,m in pickle.load((folder/'cooked.pkl').open('rb'))]
        report.setdefault(y,{})[version]=dict(parts=len(parts),bounds_m=trimesh.util.concatenate(parts).bounds.tolist(),
            watertight_count=sum(bool(m.is_watertight) for m in parts),minimum_volume_m3=float(min(m.volume for m in parts)))
        ax=axes[row,col]
        for part in parts:
            seg=trimesh.intersections.mesh_plane(part,normal,o)
            if len(seg):
                pts=np.unique((seg.reshape(-1,3)-o)@basis,axis=0)*1000
                if len(pts)>2:
                    try:
                        poly=pts[ConvexHull(pts).vertices]
                        ax.fill(poly[:,0],poly[:,1],color='#ed6b47' if col==0 else '#259c85',alpha=.35,lw=.3)
                    except Exception:pass
        seg=trimesh.intersections.mesh_plane(mesh,normal,o)
        ax.add_collection(LineCollection((seg-o)@basis*1000,colors='#142f48',linewidths=.9))
        points=(mesh.vertices-o)@basis*1000
        ax.set_xlim(points[:,0].min()-5,points[:,0].max()+5);ax.set_ylim(points[:,1].min()-5,points[:,1].max()+5)
        ax.set_aspect('equal');ax.set_title(f'{title} / {version}')
        ax.set_xlabel('mm');ax.set_ylabel('mm')
fig.suptitle('Dark lines: scanned surface | filled regions: actual PhysX collision solids')
fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(out/'collision-comparison.png',dpi=160)
(out/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
