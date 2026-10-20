"""Ordinary PNG showing contact-travel errors and collision/source cross-sections."""
import json
import numpy as np,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from audit_pitcher_contacts import ROOT,ASSET,cooked_boundary
out=ROOT/'runs/pitcher-handle-audit'
before=json.loads((out/'before/contact-audit.json').read_text())
primary=out/'coarse' if (out/'coarse/contact-audit.json').exists() else out/'after'
after=json.loads((primary/'contact-audit.json').read_text())
corrected_asset=ROOT/after['candidate']
corrected_label='Corrected: 4 mm handle slabs' if primary.name=='coarse' else 'Corrected: 1 mm handle slabs'
fig=plt.figure(figsize=(15,10));gs=fig.add_gridspec(2,2,height_ratios=[1,1.25]);ax=fig.add_subplot(gs[0,:])
for report,offset,color,label in [(before,-.2,'#dc8251','Previous collision-v2'),(after,.2,'#259c85',corrected_label)]:
    vals=[];missing=[]
    for r in report['references']:
        d=r['comparison']['candidate'];v=[abs(x)*1000 for x in d['finger_q_delta_m'] if x is not None]
        vals.append(max(v,default=0));missing.append(d['missing_contact_disagreement'])
    ax.bar(np.arange(1,41)+offset,vals,width=.4,color=color,label=label)
    for i,m in enumerate(missing):
        if m:ax.text(i+1+offset,vals[i]+.3,'x',ha='center',color=color,fontsize=9)
ax.axhline(2,color='#8d3333',linestyle='--',lw=1,label='2 mm screening threshold')
ax.set_xticks(range(1,41));ax.tick_params(axis='x',labelsize=8)
ax.set_xlabel('Original reference ID (pitcher-ref...)');ax.set_ylabel('Max per-finger travel difference (mm)')
ax.set_title('First contact compared with source scan; x = contact presence disagrees');ax.legend(fontsize=9)
meshes=[(trimesh.load(ASSET/'google_16k/textured.obj',force='mesh',process=False),'#203648','Source scan'),
    (cooked_boundary(ASSET/'collision-v2/cooked.pkl'),'#dc8251','Previous collision-v2'),
    (cooked_boundary(corrected_asset/'cooked.pkl'),'#259c85',corrected_label)]
u=np.array([.071,.069,0.]);u/=np.linalg.norm(u)
for col,(origin,basis,title) in enumerate([
    ([-.015,.03,0],np.stack([u,[0,0,1]],axis=1),'Vertical section through handle'),
    ([0,0,.18],np.array([[1,0],[0,1],[0,0]]),'Horizontal section: Z = 180 mm')]):
    ax=fig.add_subplot(gs[1,col]);normal=np.cross(basis[:,0],basis[:,1]);o=np.array(origin)
    for m,color,label in meshes:
        segments=(trimesh.intersections.mesh_plane(m,normal,o)-o)@basis*1000
        ax.add_collection(LineCollection(segments,colors=color,linewidths=1.4,label=label))
    ax.autoscale();ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('mm');ax.set_ylabel('mm');ax.legend(fontsize=8)
fig.suptitle('Pitcher local collision repair | source mesh and original 40 reference poses preserved',fontsize=15)
fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'contact-repair-comparison.png',dpi=160)
print(out/'contact-repair-comparison.png')
