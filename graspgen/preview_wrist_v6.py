"""Static PNG galleries of the measured successful hold for each saved v6 reference."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh
from prepare_rl_validation import save
from preview_strict_successes import read,pose,hand_geometry,add_mesh
from select_references import ROOT,sha


def main(name):
    folder=ROOT/'resources/validated-ycb-v6'/name
    report=read(folder/'validation.json');obj=read(folder/'bank.json')['objects'][0]
    mesh_path=ROOT/obj['mesh'].removeprefix('/workspace/')
    assert sha(mesh_path)==obj['mesh_sha256']
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    shapes=[];manifest=[]
    for i,c in enumerate(report['references']):
        ref=c['ref'];e=c['approach_evidence'];row=e['result'];trace=ROOT/e['saved_trace']
        assert sha(trace)==row['sha256']
        with np.load(trace,allow_pickle=False) as data:
            x=data['observations'][-1].astype(float)
            assert data['terminated'][-1] and not data['truncated'][-1]
            assert x[-2]>=.08 and x[-1]>=1. and np.all(x[37:39]==1.)
        object_T_hand=pose(ref['position'],ref['quaternion'])@np.linalg.inv(pose(x[30:33],x[33:37]))
        assert np.max(np.abs(np.linalg.inv(object_T_hand)[:3,3]-x[27:30]))<5e-6
        fingers=x[7:9]+.04
        assert abs(fingers.sum()-row['final_gripper_width'])<1e-6
        parts=hand_geometry(fingers)
        for part in parts.values():part.apply_transform(object_T_hand)
        shapes.append(parts)
        manifest.append(dict(number=i+1,reference=ref['id'],handle=c['handle'],
            inner_outer='innerouter' in ref['id'],object_T_hand=object_T_hand.tolist(),
            finger_joints=fingers.tolist(),trace=e['saved_trace'],trace_sha256=row['sha256']))
    for tag,azimuth in [('front',-60),('reverse',120)]:
        fig=plt.figure(figsize=(18,25),facecolor='white')
        for i,(c,parts) in enumerate(zip(report['references'],shapes)):
            ax=fig.add_subplot(8,5,i+1,projection='3d',computed_zorder=True)
            add_mesh(ax,mesh,'#289e98');add_mesh(ax,parts['wrist'],'#9aa5b1',.45);add_mesh(ax,parts['fingers'],'#efa91a')
            points=np.vstack([mesh.vertices,*[p.vertices for p in parts.values()]])
            center=(points.max(0)+points.min(0))/2;radius=np.ptp(points,axis=0).max()*.51
            ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
            ax.set_box_aspect((1,1,1),zoom=1.2);ax.set_proj_type('ortho');ax.view_init(elev=25,azim=azimuth);ax.set_axis_off()
            kind='INNER / OUTER HANDLE' if 'innerouter' in c['ref']['id'] else 'HANDLE' if c['handle'] else 'OTHER'
            ax.set_title(f'{i+1:02d}  {kind}',fontsize=10,color='#176b45',pad=0)
        fig.suptitle(f'{name.upper()} | 40 verified references | {tag}',fontsize=21,y=.987)
        fig.text(.5,.967,'Measured successful hold poses | Wrist contact with target allowed | Other collisions forbidden',ha='center',fontsize=11)
        fig.text(.5,.012,'Teal: object surface    Gold: fingers    Gray: wrist/palm\nStatic reconstruction from saved validation traces; numbers map to preview-manifest.json.',ha='center',fontsize=10)
        fig.subplots_adjust(left=.005,right=.995,top=.95,bottom=.04,wspace=.01,hspace=.13)
        fig.savefig(folder/f'40-grasps-{tag}.png',dpi=150,facecolor='white');plt.close(fig)
    save(folder/'preview-manifest.json',dict(object=name,pose_source='Measured final successful hold observation',rows=manifest))
    print('SAVED_STATIC_GALLERY',folder,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('object',choices=['mug','pitcher']);a=p.parse_args();main(a.object)
