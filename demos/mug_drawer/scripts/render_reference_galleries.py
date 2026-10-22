"""One static overlay of all 40 actual policy reference targets per object.

Reference quaternions use GARMI hand coordinates (wxyz), NOT the differently
rotated GraspGen mesh_T_gripper coordinates. This renders target poses with an
open gripper, rather than claiming they are measured post-closure poses.
"""
import argparse
import hashlib
import json
import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import pyrender
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(root, path):
    return root / path.removeprefix('/workspace/')


def origin(node):
    result = np.eye(4)
    if node is not None:
        result[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        result[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return result


def hand_geometry(root):
    urdf = ET.parse(root/'rl/assets/garmi_training.urdf').getroot()
    offset = np.array(read(root/'rl/assets/manifest.json')['link7_T_hand'])
    groups = {'palm': [], 'fingers': []}
    with (root/'assets/kitchen/cooked-model.pkl').open('rb') as f:
        cooked = pickle.load(f)
    for record, mesh in cooked:
        body = record.get('body')
        if body == 'right_fr3_link7':
            tf, group = np.linalg.inv(offset), 'palm'
        elif body in ('right_fr3_leftfinger', 'right_fr3_rightfinger'):
            joint = next(j for j in urdf.findall('joint') if j.find('child').get('link') == body)
            shift = np.eye(4)
            shift[:3, 3] = np.fromstring(joint.find('axis').get('xyz'), sep=' ') * .04
            tf, group = origin(joint.find('origin')) @ shift, 'fingers'
        else:
            continue
        part = mesh.copy()
        part.apply_transform(tf)
        groups[group].append(part)
    return {k: trimesh.util.concatenate(v) for k, v in groups.items()}


def tubes(paths, radius):
    """Real 3D tubes: object surfaces and outlines share one depth buffer."""
    pieces = []
    for points in paths:
        for start, end in zip(points[:-1], points[1:]):
            pieces.append(trimesh.creation.cylinder(radius=radius, segment=[start, end], sections=10))
    return trimesh.util.concatenate(pieces)


def render_overlay(mesh, outlines, stems, name):
    width, height = 1800, 1500
    points = np.vstack([mesh.vertices, *outlines, *stems])
    center = (points.min(0)+points.max(0))/2
    azimuth, elevation = np.deg2rad([-60, 27])
    back = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
    right = np.cross([0, 0, 1], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    basis = np.column_stack([right, up, back])
    projected = (points-center) @ basis
    xy_center = (projected[:, :2].min(0)+projected[:, :2].max(0))/2
    center += basis[:, :2] @ xy_center
    spans = np.ptp(projected[:, :2], axis=0)
    xmag = max(spans[0]/2, spans[1]/2*width/height)*1.12
    pose = np.eye(4); pose[:3, :3] = basis; pose[:3, 3] = center+back*.8
    scene = pyrender.Scene(bg_color=[1., 1., 1., 1.], ambient_light=[.5, .5, .5])
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=xmag*height/width, znear=.01, zfar=5.)
    scene.add(camera, pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.), pose=pose)
    def material(rgb):
        return pyrender.MetallicRoughnessMaterial(baseColorFactor=[*rgb, 1.], metallicFactor=0., roughnessFactor=.8, doubleSided=True)
    obj = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material([.07, .48, .53]), smooth=True))
    fingers = scene.add(pyrender.Mesh.from_trimesh(tubes(outlines, .00065), material=material([.92, .43, .055]), smooth=False))
    axes = scene.add(pyrender.Mesh.from_trimesh(tubes(stems, .00025), material=material([.55, .60, .66]), smooth=False))
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        rgb, _ = renderer.render(scene)
        # Independently count outline pixels genuinely in front/behind the mesh.
        fingers.mesh.is_visible = axes.mesh.is_visible = False
        _, object_depth = renderer.render(scene)
        fingers.mesh.is_visible = axes.mesh.is_visible = True
        obj.mesh.is_visible = False
        _, gripper_depth = renderer.render(scene)
        overlap = (object_depth > 0) & (gripper_depth > 0)
        near = overlap & (gripper_depth < object_depth-1.e-5)
        far = overlap & ~near
        audit = dict(renderer='pyrender EGL; shared per-pixel depth buffer',
                     projected_overlap_pixels=int(overlap.sum()),
                     gripper_in_front_pixels=int(near.sum()), gripper_behind_pixels=int(far.sum()))
        assert near.sum() > 0 and far.sum() > 0
    finally:
        renderer.delete()
    canvas = Image.new('RGB', (1800, 1800), 'white')
    canvas.paste(Image.fromarray(rgb), (0, 150))
    pen = ImageDraw.Draw(canvas)
    font_dir = '/usr/share/fonts/truetype/dejavu'
    title = ImageFont.truetype(font_dir+'/DejaVuSans-Bold.ttf', 44)
    caption = ImageFont.truetype(font_dir+'/DejaVuSans.ttf', 24)
    pen.text((900, 65), f'{name.upper()}  |  40 REFERENCE GRASPS', font=title, anchor='mm', fill='#253748')
    pen.text((900, 1705), 'All 40 target poses overlaid  |  Open grippers (80 mm)', font=caption, anchor='mm', fill='#526476')
    pen.text((900, 1745), 'Orange: fingers / bridge   |   Gray: approach axis   |   True 3D occlusion', font=caption, anchor='mm', fill='#526476')
    return canvas, audit


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mug-workspace', type=Path, required=True)
    p.add_argument('--bowl-workspace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mug_bank = args.mug_workspace/'resources/validated-ycb-v6/mug/bank.json'
    bowl_bank = args.bowl_workspace/'results/cram-rl-integration/policy-7500/reference-bank.json'
    profile = yaml.safe_load((args.mug_workspace/'runs/mug-physical-success-review/ppo.yaml').read_text())
    allowed = profile['garmi_task']['multi_object']['reference_ids']['mug']
    hand = hand_geometry(args.mug_workspace)
    provenance = {'display': 'Exact reference target pose; hand frame; 80 mm open gripper',
                  'rendering': 'True 3D depth; no painter-order mesh overlay', 'objects': []}
    for name, root, bank in [('mug', args.mug_workspace, mug_bank), ('bowl', args.bowl_workspace, bowl_bank)]:
        obj = next(o for o in read(bank)['objects'] if o['name'] == name)
        refs = obj['references']
        assert len(refs) == len({r['id'] for r in refs}) == 40
        if name == 'mug':
            assert set(allowed) == {r['id'] for r in refs}
        mesh_path = resolve(root, obj['mesh'])
        assert digest(mesh_path) == obj['mesh_sha256']
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        # Original scale and geometry: no reconstruction or coordinate recentering.
        poses = []
        for ref in refs:
            tf = np.eye(4)
            q = np.array(ref['quaternion'])
            tf[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            tf[:3, 3] = ref['position']
            poses.append(tf)
        record = dict(object=name, bank=str(bank), bank_sha256=digest(bank), mesh_sha256=digest(mesh_path),
                      count=40, references=[dict(id=r['id']) for r in refs])
        provenance['objects'].append(record)
        # Compact U-shaped line drawings retain all reference positions and
        # orientations without covering the object with 40 solid palms.
        fingers = hand['fingers'].bounds
        zbase, ztip = fingers[0, 2], fingers[1, 2]
        local = np.array([[0, -.04, ztip], [0, -.04, zbase],
                          [0, .04, zbase], [0, .04, ztip]])
        stem = np.array([[0, 0, zbase], [0, 0, .015]])
        outlines, stems = [], []
        for tf in poses:
            outlines.append(trimesh.transform_points(local, tf))
            stems.append(trimesh.transform_points(stem, tf))
        canvas, audit = render_overlay(mesh, outlines, stems, name)
        record['depth_audit'] = audit
        target = args.out/f'{name}-all-40.png'
        canvas.save(target)
        print('OVERLAY', target, flush=True)
    (args.out/'reference-provenance.json').write_text(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
