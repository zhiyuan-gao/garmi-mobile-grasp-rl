"""Static inner/outer handle candidates; purple inner finger, orange outer finger."""
import argparse,math,json
import numpy as np,trimesh,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from select_references import ROOT,HAND_OFFSET
from prepare_rl_validation import hand_parts
from preview_strict_successes import add_mesh
from try_handle_roll import host,read
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--passed-only',action='store_true');a=p.parse_args();folder=ROOT/a.folder
obj=read(folder/'bank.json')['objects'][0];refs=obj['references'];name=obj['name']
mesh=trimesh.load(host(obj['mesh']),force='mesh',process=False)
summary=read(folder/'endpoint/summary.json') if (folder/'endpoint/summary.json').exists() else None
if a.passed_only:
 assert summary is not None
 refs=[r for r in refs if summary['per_reference'][r['id']]['successes']>0]
 assert refs
for tag in ['front','angled']:
 cols=min(3,len(refs));rows=math.ceil(len(refs)/cols);fig=plt.figure(figsize=(5.5*cols,4.5*rows+.7))
 for index,ref in enumerate(refs):
  h=np.array(ref['mesh_T_gripper'])@HAND_OFFSET
  parts=hand_parts();qs=[c['q_m'] for c in ref['contacts']['cooked']]
  for k,part in enumerate(parts):
   if 2<=k<=5:part.apply_translation([0,qs[0]-.04,0])
   if 6<=k<=9:part.apply_translation([0,-(qs[1]-.04),0])
   part.apply_transform(h)
  ax=fig.add_subplot(rows,cols,index+1,projection='3d');add_mesh(ax,mesh,'#289e98')
  for k,part in enumerate(parts):add_mesh(ax,part,'#9aa5b1' if k<2 else '#efa91a' if k<6 else '#a855d1',.28 if k<2 else 1.)
  allp=np.vstack([mesh.vertices,*[m.vertices for m in parts]])
  center=(allp.max(0)+allp.min(0))/2;radius=np.ptp(allp,axis=0).max()*.51
  ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
  ax.set_box_aspect((1,1,1));ax.set_axis_off()
  azimuth=134 if name=='pitcher' else 90
  ax.view_init(elev=8 if tag=='front' else 25,azim=azimuth+(0 if tag=='front' else -35))
  result='pending physics' if summary is None else str(summary['per_reference'][ref['id']]['successes'])+'/'+str(summary['per_reference'][ref['id']]['attempts'])+' endpoint passes'
  ax.set_title(ref['id']+'\n'+result,fontsize=9)
 fig.suptitle(name.title()+' | purple: inner finger; orange: outer finger\nGeometric first-contact pose\n(Not a measured successful-hold image)',fontsize=10)
 prefix='passed-candidate' if a.passed_only else 'inner-outer'
 fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(folder/f'{prefix}-{tag}.png',dpi=150);plt.close(fig)
