"""Normal PNG previews of only the references retained by local repeatability."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from select_references import ROOT,HAND_OFFSET
from prepare_rl_validation import hand_parts


def main(folder):
    obj=json.loads((folder/'bank.json').read_text())['objects'][0]
    refs=obj['references'];n=len(refs)
    mesh=trimesh.load(ROOT/obj['mesh'].removeprefix('/workspace/'),force='mesh',process=False)
    hand=trimesh.util.concatenate(hand_parts());hand.apply_transform(HAND_OFFSET)
    points,_=trimesh.sample.sample_surface(mesh,3500,seed=42)
    conditions=json.loads((folder/'endpoint-conditions.json').read_text())['conditions']
    preparation=json.loads((folder/'preparation.json').read_text())
    panels=[(r,-60) for r in refs] if n>1 else [(refs[0],a) for a in (-60,30,120)]
    columns=min(7,len(panels));rows=math.ceil(len(panels)/columns)
    fig=plt.figure(figsize=(3*columns,3.5*rows+1),facecolor='white')
    for i,(ref,azim) in enumerate(panels):
        ax=fig.add_subplot(rows,columns,i+1,projection='3d')
        g=np.array(ref['mesh_T_gripper']);hv=trimesh.transform_points(hand.vertices,g)
        ax.scatter(*points.T,s=.6,c='#188e78',alpha=.45,depthshade=False)
        ax.add_collection3d(Poly3DCollection(hv[hand.faces],facecolors='#e6af31',linewidths=0,alpha=.85))
        v=np.vstack([mesh.vertices,hv]);center=(v.min(0)+v.max(0))/2;radius=np.ptp(v,axis=0).max()*.54
        ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
        ax.set_box_aspect((1,1,1));ax.view_init(23,azim);ax.set_axis_off()
        accepted=[c for c in conditions if c['reference']==ref['id']]
        label=f"{ref['id']}\n{'handle' if accepted[0]['handle'] else 'other'} | {len(accepted)} accepted condition(s)"
        if preparation.get('object_window_halfwidth_m') is None:
            radii=sorted({c['validated_object_offset_radius_m']*1000 for c in accepted})
            label+=' | '+', '.join(f'+/-{r:g} mm' for r in radii)
        ax.set_title(label,fontsize=9,color='#176b45',y=.95)
    target=preparation.get('target_reference_count')
    incomplete=f' (target {target} not reached)' if target and n<target else ''
    fig.suptitle(f"{obj['name']}: {n} retained grasp(s){incomplete} | yellow: open GARMI hand/wrist",fontsize=14)
    radius=preparation.get('object_window_halfwidth_m')
    footer=(f'Only listed local object windows (+/-{radius*1000:g} mm); fixed tested orientation.' if radius is not None
            else 'Per-reference local windows are shown in the panels; fixed tested orientation.')
    fig.text(.5,.02,footer+' Expert approach remains untested.',ha='center',fontsize=10)
    fig.subplots_adjust(left=0,right=1,bottom=.08,top=.88,wspace=-.05,hspace=.1)
    fig.savefig(folder/'retained-grasps.png',dpi=160);plt.close(fig)
    print(folder/'retained-grasps.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);a=p.parse_args();main(a.folder)
