"""Ordinary PNGs of the original scan, repaired sampling surface and point cloud."""
from pathlib import Path
import numpy as np,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/mug-surface-v3'
old=trimesh.load(ROOT/'assets/ycb/025_mug/google_16k/textured.obj',force='mesh',process=False)
new=trimesh.load(ROOT/'assets/ycb/025_mug/surface-v3/repaired.obj',force='mesh',process=False)
fig=plt.figure(figsize=(13,8))
for i,(m,title,color) in enumerate([(old,'Original scan','#b85c4b'),(new,'Repaired sampling surface','#259c85')]):
    ax=fig.add_subplot(2,3,i+1,projection='3d')
    ax.add_collection3d(Poly3DCollection(m.triangles,facecolors=color,linewidths=0,alpha=1,shade=True))
    ax.set_xlim(-.075,.065);ax.set_ylim(-.05,.09);ax.set_zlim(-.005,.095)
    ax.set_box_aspect((1.4,1.4,1));ax.view_init(elev=48,azim=-65);ax.set_axis_off();ax.set_title(title)
ax=fig.add_subplot(2,3,3,projection='3d')
clouds=out/'025_mug/input-pointclouds.npz'
pc=np.load(clouds)['points'][0] if clouds.exists() else trimesh.sample.sample_surface(new,2048,seed=42)[0]
ax.scatter(*pc.T,s=1.8,c=pc[:,2],cmap='viridis');ax.set_xlim(-.075,.065);ax.set_ylim(-.05,.09);ax.set_zlim(-.005,.095)
ax.set_box_aspect((1.4,1.4,1));ax.view_init(elev=48,azim=-65);ax.set_axis_off();ax.set_title('GraspGen input: 2,048 points' if clouds.exists() else 'New surface: 2,048 sampled points')
for j,(normal,origin,columns,title) in enumerate([
    ([0,1,0],[0,.015,0],[0,2],'Vertical: cavity and handle'),
    ([0,0,1],[0,0,.020],[0,1],'Horizontal: Z = 20 mm'),
    ([1,0,0],[.0195,0,0],[1,2],'Handle roots: X = 19.5 mm')]):
    ax=fig.add_subplot(2,3,j+4)
    for m,color,label in [(old,'#b85c4b','Original'),(new,'#188e78','Repaired')]:
        seg=trimesh.intersections.mesh_plane(m,normal,origin)[:,:,columns]*1000
        ax.add_collection(LineCollection(seg,colors=color,linewidths=1.1,label=label))
    ax.autoscale();ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('mm');ax.set_ylabel('mm');ax.legend(fontsize=8)
fig.suptitle('Mug surface repair: hollow body, connected handle, no internal partition faces')
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'surface-and-pointcloud.png',dpi=170)
print(out/'surface-and-pointcloud.png')
