"""Static gallery of repaired-mug references, with physical diagnostic outcomes."""
from pathlib import Path
import json
import numpy as np,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from prepare_rl_validation import hand_parts
from select_references import ROOT,HAND_OFFSET
out=ROOT/'runs/mug-surface-v3/025_mug'
data=np.load(out/'selected-40.npz');poses=data['T_object_graspgen'];scores=data['scores']
refs=json.loads((out/'selected-40.json').read_text())
mesh=trimesh.load(ROOT/'assets/ycb/025_mug/surface-v3/repaired.obj',force='mesh',process=False)
hand=trimesh.util.concatenate(hand_parts());hand.apply_transform(HAND_OFFSET)
pc,_=trimesh.sample.sample_surface(mesh,2500,seed=42)
physics=json.loads((out/'physics/summary.json').read_text()) if (out/'physics/summary.json').exists() else None
fig=plt.figure(figsize=(20,15),facecolor='white')
for j,g in enumerate(poses):
    ax=fig.add_subplot(5,8,j+1,projection='3d');hv=trimesh.transform_points(hand.vertices,g)
    ax.scatter(*pc.T,s=.35,c='#188e78',alpha=.4,depthshade=False)
    ax.add_collection3d(Poly3DCollection(hv[hand.faces],facecolors='#e6af31',linewidths=0,alpha=.8))
    v=np.vstack([mesh.vertices,hv]);c=(v.min(0)+v.max(0))/2;r=np.ptp(v,axis=0).max()*.54
    ax.set_xlim(c[0]-r,c[0]+r);ax.set_ylim(c[1]-r,c[1]+r);ax.set_zlim(c[2]-r,c[2]+r)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-60);ax.set_axis_off()
    kind='handle' if j<10 else 'other';label=f'{j+1:02d} | {kind} | {scores[j]:.2f}';color='#333333'
    if physics:
        p=physics['per_reference'][refs[j]['id']];label+=f"\nLift tests: {p['successes']}/{p['attempts']}"
        color='#17734b' if p['successes'] else '#b03c32'
    ax.set_title(label,fontsize=9,y=.95,color=color)
fig.suptitle('Repaired mug | 40 NEW references (mug-v3-ref001...040) | yellow: actual open hand/wrist colliders',fontsize=16)
fig.text(.5,.01,'Green: sampled repaired surface. Tests initialize IK directly; no approach trajectories or training.',ha='center',fontsize=12)
fig.subplots_adjust(left=0,right=1,bottom=.035,top=.94,wspace=-.08,hspace=.08)
fig.savefig(out/'gallery-40.png',dpi=160);plt.close(fig)
print(out/'gallery-40.png')
