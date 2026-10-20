"""Paired static geometric previews, explicitly not physical-success pictures."""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh
from select_references import ROOT,HAND_OFFSET
from rebalance_mug import FingerContacts
from preview_strict_successes import hand_geometry,add_mesh
from try_handle_roll import OUT,host,read

for name in ['mug','pitcher']:
    folder=OUT/name;bank=read(folder/'bank.json')['objects'][0];refs=bank['references']
    source=ROOT/f'resources/validated-ycb-v{4 if name=="mug" else 5}/{name}/bank.json'
    parents={r['id']:r for r in read(source)['objects'][0]['references']}
    mesh=trimesh.load(host(bank['mesh']),force='mesh',process=False)
    checker=FingerContacts(mesh)
    for view,azimuth in [('front',-60),('reverse',120)]:
        fig=plt.figure(figsize=(11,3.8*len(refs)+.7))
        for i,r in enumerate(refs):
            for col,item in enumerate([parents[r['parent_reference']],r]):
                t=np.array(item['mesh_T_gripper']);hit=checker.query(t)
                shapes=hand_geometry([c['q_m'] if c else .04 for c in hit])
                ax=fig.add_subplot(len(refs),2,2*i+col+1,projection='3d')
                add_mesh(ax,mesh,'#289e98')
                for k,m in shapes.items():
                    m.apply_transform(t@HAND_OFFSET)
                    add_mesh(ax,m,'#efa91a' if k=='fingers' else '#9aa5b1',1 if k=='fingers' else .35)
                pts=np.vstack([mesh.vertices,*[m.vertices for m in shapes.values()]])
                c=(pts.max(0)+pts.min(0))/2;radius=np.ptp(pts,axis=0).max()*.52
                ax.set(xlim=(c[0]-radius,c[0]+radius),ylim=(c[1]-radius,c[1]+radius),zlim=(c[2]-radius,c[2]+radius))
                ax.set_box_aspect((1,1,1));ax.view_init(elev=25,azim=azimuth);ax.set_axis_off()
                label='Parent reference' if col==0 else f'Local Z {r["local_z_roll_degrees"]:+d} deg + translation'
                ax.set_title(label+'\n'+item['id'],fontsize=10)
        fig.suptitle(name.title()+' handle roll comparison\nGeometry at independent first finger contacts; NOT measured successful holds',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(folder/f'roll90-comparison-{view}.png',dpi=150);plt.close(fig)
