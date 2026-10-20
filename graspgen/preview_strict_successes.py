"""Static galleries of measured successful hold poses, reconstructed from saved observations."""
import hashlib
import json
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from select_references import ROOT, origin

OUT = ROOT/'runs/ycb-expert-precision-v1'


def read(path):
    return json.loads(path.read_text())


def host(path):
    return ROOT/path.removeprefix('/workspace/')


def pose(position, quaternion):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = position
    return transform


def hand_geometry(finger_positions):
    urdf = ET.parse(ROOT/'rl/assets/garmi_training.urdf').getroot()
    link7_T_hand = np.array(read(ROOT/'rl/assets/manifest.json')['link7_T_hand'])
    groups = {'wrist': [], 'fingers': []}
    for record, mesh in pickle.load((ROOT/'assets/kitchen/cooked-model.pkl').open('rb')):
        body = record.get('body')
        if body == 'right_fr3_link7':
            transform = np.linalg.inv(link7_T_hand)
            group = 'wrist'
        elif body in ('right_fr3_leftfinger', 'right_fr3_rightfinger'):
            index = 0 if body == 'right_fr3_leftfinger' else 1
            joint = next(j for j in urdf.findall('joint') if j.find('child').get('link') == body)
            transform = origin(joint.find('origin'))
            shift = np.eye(4)
            shift[:3, 3] = np.fromstring(joint.find('axis').get('xyz'), sep=' ')*finger_positions[index]
            transform = transform@shift
            group = 'fingers'
        else:
            continue
        part = mesh.copy()
        part.apply_transform(transform)
        groups[group].append(part)
    return {key: trimesh.util.concatenate(parts) for key, parts in groups.items()}


def add_mesh(ax, mesh, color, alpha=1.):
    ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=color,
        linewidths=0, alpha=alpha, shade=True, lightsource=LightSource(azdeg=300, altdeg=45)))


def main(name):
    folder = OUT/name
    summary = read(folder/'strict/summary.json')
    assert summary['status'] == 'complete'
    successful = {row['reference']: row for row in summary['results'] if row['reason'] == 'success'}
    config = yaml.safe_load((folder/'strict.yaml').read_text())
    obj = next(o for o in read(host(config['garmi_task']['multi_object']['manifest']))['objects'] if o['name'] == name)
    refs = [r for r in obj['references'] if r['id'] in successful]
    assert len(refs) == len(successful) == summary['counts']['success']
    conditions = read(host(config['expert_preflight']['endpoint_conditions']))['conditions']
    handle = {c['reference']: c['handle'] for c in conditions}
    mesh_path = host(obj['mesh'])
    assert hashlib.sha256(mesh_path.read_bytes()).hexdigest() == obj['mesh_sha256']
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    gallery = folder/'strict-success-preview'
    gallery.mkdir(exist_ok=True)
    manifest = dict(object=name, count=len(refs), handle_count=sum(handle[r['id']] for r in refs),
        source_summary=str(folder.relative_to(ROOT)/'strict/summary.json'),
        source_summary_sha256=hashlib.sha256((folder/'strict/summary.json').read_bytes()).hexdigest(),
        mesh=str(mesh_path.relative_to(ROOT)), mesh_sha256=obj['mesh_sha256'],
        pose_source='Final saved observation at physical success; object-frame hand pose and measured finger joints',
        scope='One successful trial per displayed reference; not a simulator screenshot or a repeatability claim',
        rows=[])
    shapes = []
    for number, ref in enumerate(refs, 1):
        row = successful[ref['id']]
        trajectory = host(row['file'])
        assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == row['sha256']
        with np.load(trajectory, allow_pickle=False) as saved:
            x = saved['observations'][-1].astype(float)
            assert saved['terminated'][-1] and not saved['truncated'][-1]
            assert x[-2] >= .08 and x[-1] >= 1. and np.all(x[37:39] == 1.)
            final_phase = int(saved['phases'][-1])
        object_T_goal = pose(ref['position'], ref['quaternion'])
        hand_T_goal = pose(x[30:33], x[33:37])
        object_T_hand = object_T_goal@np.linalg.inv(hand_T_goal)
        # Independent object-origin observation checks the reconstruction/frame convention.
        frame_error = float(np.max(np.abs(np.linalg.inv(object_T_hand)[:3, 3]-x[27:30])))
        assert frame_error < 5.e-6, frame_error
        finger_positions = x[7:9]+.04
        assert abs(finger_positions.sum()-row['final_gripper_width']) < 1.e-6
        parts = hand_geometry(finger_positions)
        for part in parts.values():
            part.apply_transform(object_T_hand)
        shapes.append(parts)
        manifest['rows'].append(dict(number=number, reference=ref['id'], handle=handle[ref['id']],
            trajectory=str(trajectory.relative_to(ROOT)), trajectory_sha256=row['sha256'],
            object_T_hand=object_T_hand.tolist(), finger_joint_positions=finger_positions.tolist(),
            frame_error_m=frame_error, phase=final_phase, lift_m=row['height'], slip_m=row['slip']))

    for tag, azimuth in [('front', -60), ('reverse', 120)]:
        cols = 4
        rows = math.ceil(len(refs)/cols)
        fig = plt.figure(figsize=(16, 3.15*rows+1.1), facecolor='white')
        for i, (ref, parts) in enumerate(zip(refs, shapes)):
            ax = fig.add_subplot(rows, cols, i+1, projection='3d', computed_zorder=True)
            add_mesh(ax, mesh, '#289e98')
            add_mesh(ax, parts['wrist'], '#9aa5b1', .50)
            add_mesh(ax, parts['fingers'], '#efa91a')
            points = np.vstack([mesh.vertices, *[p.vertices for p in parts.values()]])
            center = (points.max(0)+points.min(0))/2
            radius = np.ptp(points, axis=0).max()*.51
            ax.set_xlim(center[0]-radius, center[0]+radius)
            ax.set_ylim(center[1]-radius, center[1]+radius)
            ax.set_zlim(center[2]-radius, center[2]+radius)
            ax.set_box_aspect((1, 1, 1), zoom=1.23)
            ax.set_proj_type('ortho')
            ax.view_init(elev=25, azim=azimuth)
            ax.set_axis_off()
            kind = 'HANDLE' if handle[ref['id']] else 'OTHER'
            ax.set_title(f"{i+1:02d}  {ref['id']}\n{kind} | successful hold", fontsize=10,
                         color='#176b45', pad=0)
        fig.suptitle(f"{name.upper()} | {len(refs)} successful grasps | strict settings", fontsize=19, y=.985)
        fig.text(.5, .955, f"Measured hold pose | {manifest['handle_count']} handle, {len(refs)-manifest['handle_count']} other | {tag} view",
                 ha='center', fontsize=12, color='#45515e')
        fig.text(.5, .018, 'Teal: object surface   Gold: fingers   Gray: wrist/palm (transparent)\n'
                 'Reconstructed from saved successful trajectories; object frame. One tested scene per reference.',
                 ha='center', fontsize=10, color='#45515e')
        fig.subplots_adjust(left=.005, right=.995, top=.92, bottom=.065, wspace=.015, hspace=.16)
        target = gallery/f'{name}-strict-successes-{tag}.png'
        fig.savefig(target, dpi=170, facecolor='white')
        plt.close(fig)
        print(target, flush=True)
    (gallery/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    for object_name in ('mug', 'pitcher'):
        main(object_name)
