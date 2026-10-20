"""Static three-view coverage of validated TCP positions and approach directions."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from select_references import ROOT


def main(folder):
    obj=json.loads((folder/'bank.json').read_text())['objects'][0]
    refs=obj['references'];conditions=json.loads((folder/'endpoint-conditions.json').read_text())['conditions']
    handle={c['reference']:c['handle'] for c in conditions}
    mesh=trimesh.load(ROOT/obj['mesh'].removeprefix('/workspace/'),force='mesh',process=False)
    points,_=trimesh.sample.sample_surface(mesh,4500,seed=43)
    transforms=np.array([r['mesh_T_gripper'] for r in refs])
    centers=transforms[:,:3,3]+transforms[:,:3,2]*.0934
    start=centers-transforms[:,:3,2]*.025
    colors=['#a85d19' if handle[r['id']] else '#235f9b' for r in refs]
    vertices=np.vstack([mesh.vertices,centers,start]);middle=(vertices.min(0)+vertices.max(0))/2
    radius=np.ptp(vertices,axis=0).max()*.54
    fig=plt.figure(figsize=(13,4.6),layout='constrained')
    for i,azimuth in enumerate((-60,60,180)):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        ax.scatter(*points.T,s=.8,c='#b7bebc',alpha=.2,depthshade=False)
        for a,b,c in zip(start,centers,colors):ax.plot(*np.vstack([a,b]).T,color=c,alpha=.7,lw=1.1)
        ax.scatter(*centers.T,c=colors,s=20,depthshade=False)
        ax.set_xlim(middle[0]-radius,middle[0]+radius);ax.set_ylim(middle[1]-radius,middle[1]+radius)
        ax.set_zlim(middle[2]-radius,middle[2]+radius);ax.set_box_aspect((1,1,1));ax.view_init(25,azimuth)
        ax.set_xlabel('X (m)',fontsize=8);ax.set_ylabel('Y (m)',fontsize=8);ax.set_zlabel('Z (m)',fontsize=8)
        ax.tick_params(labelsize=7);ax.set_title(f'View {i+1}',fontsize=10)
    count=sum(handle[r['id']] for r in refs)
    fig.suptitle(f"{obj['name']}: {len(refs)} validated references | dots: grasp centers, lines: approach directions",fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color='#a85d19',marker='o',label=f'Handle: {count}'),
        Line2D([0],[0],color='#235f9b',marker='o',label=f'Other: {len(refs)-count}')],loc='lower center',ncol=2,fontsize=9)
    fig.savefig(folder/'coverage.png',dpi=170,bbox_inches='tight',pad_inches=.18);plt.close(fig)
    print(folder/'coverage.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);main(p.parse_args().folder)
