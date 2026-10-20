"""Render all 40 selected references, including an offline interactive selector."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import plotly.graph_objects as go
import trimesh

from select_references import ROOT, OBJECTS, hand_geometry, pose_components


def mesh_trace(mesh, color, name, visible=True):
    v, f = mesh.vertices, mesh.faces
    return go.Mesh3d(x=v[:,0], y=v[:,1], z=v[:,2], i=f[:,0], j=f[:,1], k=f[:,2],
        name=name, color=color, visible=visible, opacity=1.)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/graspgen-ycb-40refs')
    args = parser.parse_args()
    hand = hand_geometry()
    for ycb, name in OBJECTS.items():
        out = args.out/ycb
        refs = json.loads((out/'selected-40.json').read_text())
        data = np.load(out/'selected-40.npz')
        poses, scores = data['T_object_graspgen'], data['scores']
        mesh = trimesh.load(ROOT/'assets/ycb'/ycb/'google_16k/textured.obj', force='mesh', process=False)
        color = '#416ca6' if name == 'pitcher' else '#ad5047'
        traces = [mesh_trace(mesh, color, name)]
        for j, g in enumerate(poses):
            moved = hand.copy(); moved.apply_transform(g)
            traces.append(mesh_trace(moved, '#e8b135', refs[j]['id'], j == 0))
        buttons = []
        for j, ref in enumerate(refs):
            buttons.append(dict(label=f"{ref['id']} | score {ref['score']:.3f}", method='update',
                args=[dict(visible=[True]+[i==j for i in range(40)])]))
        fig = go.Figure(traces)
        fig.update_layout(title=f'{name}: 40 diverse references | GARMI hand geometry | no physical lift validation',
            scene=dict(aspectmode='data', xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'),
            updatemenus=[dict(buttons=buttons, x=0, y=1.12)], margin=dict(l=0,r=0,b=0,t=115))
        fig.write_html(out/'viewer-40.html', include_plotlyjs=True)

        # Contact-center locations and approach directions at their actual object coordinates.
        centers, rotations = pose_components(poses)
        fig = plt.figure(figsize=(12,5.5), facecolor='white')
        for k, az in enumerate([-60,120]):
            ax = fig.add_subplot(1,2,k+1,projection='3d')
            ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=color, linewidths=0, alpha=.15))
            colors = plt.cm.turbo(np.linspace(0,1,40))
            ax.scatter(*centers.T, c=colors, s=32, depthshade=False)
            for j, p in enumerate(centers):
                ax.text(*p, str(j+1), fontsize=7)
            direction = -rotations[:,:,2]*.035
            ax.quiver(*centers.T, *direction.T, colors=colors, linewidth=.8, arrow_length_ratio=.15)
            v = np.vstack([mesh.vertices, centers, centers+direction])
            c = (v.min(0)+v.max(0))/2; r=np.ptp(v,axis=0).max()*.55
            ax.set_xlim(c[0]-r,c[0]+r); ax.set_ylim(c[1]-r,c[1]+r); ax.set_zlim(c[2]-r,c[2]+r)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=az);ax.set_axis_off()
        fig.suptitle(f'{ycb}: 40 selected reference locations and orientations', fontsize=16)
        fig.text(.5,.02,'Numbers = reference IDs. Points = hand TCP; segments extend back toward the pregrasp.',ha='center',fontsize=10)
        fig.subplots_adjust(left=0,right=1,bottom=.05,top=.93,wspace=0)
        fig.savefig(out/'coverage-40.png',dpi=160);plt.close(fig)

        # Point-cloud object display keeps the full 40-panel gallery compact and fast.
        pc,_ = trimesh.sample.sample_surface(mesh,1800,seed=42)
        fig = plt.figure(figsize=(20,14), facecolor='white')
        for j,g in enumerate(poses):
            ax=fig.add_subplot(5,8,j+1,projection='3d')
            hv=trimesh.transform_points(hand.vertices,g)
            ax.scatter(*pc.T,s=.35,c=color,alpha=.32,depthshade=False)
            ax.add_collection3d(Poly3DCollection(hv[hand.faces],facecolors='#e6af31',linewidths=0,alpha=.9))
            v=np.vstack([mesh.vertices,hv]);c=(v.min(0)+v.max(0))/2;r=np.ptp(v,axis=0).max()*.54
            ax.set_xlim(c[0]-r,c[0]+r);ax.set_ylim(c[1]-r,c[1]+r);ax.set_zlim(c[2]-r,c[2]+r)
            ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-60);ax.set_axis_off()
            ax.set_title(f'{j+1:02d} | score {scores[j]:.3f}',fontsize=10,y=.95)
        fig.suptitle(f'{ycb} | all 40 diverse references | yellow = open GARMI hand colliders',fontsize=18)
        fig.text(.5,.01,'Original scale. Geometry screened; not validated for arm reachability, finger closure, or lift.',ha='center',fontsize=12)
        fig.subplots_adjust(left=0,right=1,bottom=.035,top=.95,wspace=-.08,hspace=.05)
        fig.savefig(out/'gallery-40.png',dpi=160);plt.close(fig)
        print(out,flush=True)


if __name__=='__main__':
    main()
