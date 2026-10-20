"""Choose 40 spatially diverse, geometry-screened references per YCB object."""
import argparse
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from trimesh.collision import CollisionManager

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = {'019_pitcher_base': 'pitcher', '025_mug': 'mug'}
HAND_OFFSET = np.eye(4)
HAND_OFFSET[:3, :3] = Rotation.from_euler('z', -90, degrees=True).as_matrix()
SETTINGS = dict(count=40, score_threshold=.5, table_clearance_m=.002,
    pregrasp_distance_m=.08, approach_sample_step_m=.005,
    position_scale_m=.03, rotation_scale_degrees=30., finger_opening_m=.08,
    contact_center_hand_m=[0., 0., .0934], symmetric_finger_swap=True)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_bowl_conversion():
    refs = json.loads((ROOT/'resources/bowl-40-references.json').read_text())
    assert len(refs) == 40
    for ref in refs:
        t = np.asarray(ref['mesh_T_gripper']) @ HAND_OFFSET
        q = np.array(ref['quaternion'])
        assert np.allclose(t[:3, :3], Rotation.from_quat(q[[1,2,3,0]]).as_matrix(), atol=3e-7)
        assert np.allclose(t[:3, 3], ref['position'], atol=1e-8)


def origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def hand_geometry():
    """Use this project's GARMI palm and finger collision primitives, in GraspGen frame."""
    stored = ROOT/'assets/graspgen/garmi-hand.stl'
    if not stored.exists():
        source = ROOT.parent/'cram-vrb-lab/assets/garmi_description/garmi_description/meshes/hand.stl'
        stored.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, stored)
    urdf = ET.parse(ROOT/'rl/assets/garmi_training.urdf').getroot()
    parts = []
    for link_name in ('right_fr3_hand', 'right_fr3_leftfinger', 'right_fr3_rightfinger'):
        link = urdf.find(f"link[@name='{link_name}']")
        t = np.eye(4)
        if link_name != 'right_fr3_hand':
            joint = next(j for j in urdf.findall('joint') if j.find('child').get('link') == link_name)
            t = origin(joint.find('origin'))
            shift = np.eye(4)
            shift[:3, 3] = np.fromstring(joint.find('axis').get('xyz'), sep=' ')*SETTINGS['finger_opening_m']/2
            t = t @ shift
        for collision in link.findall('collision'):
            geo = collision.find('geometry')
            if geo.find('box') is not None:
                mesh = trimesh.creation.box(np.fromstring(geo.find('box').get('size'), sep=' '))
            elif geo.find('mesh') is not None:
                mesh = trimesh.load(stored, force='mesh')
            else:
                raise ValueError('Unsupported hand collider')
            mesh.apply_transform(HAND_OFFSET @ t @ origin(collision.find('origin')))
            parts.append(mesh)
    return trimesh.util.concatenate(parts)


def pose_components(poses):
    rotations = poses[:, :3, :3]
    # TCP/contact-center positions avoid coupling wrist displacement to orientation.
    centers = poses[:, :3, 3] + rotations @ np.array(SETTINGS['contact_center_hand_m'])
    return centers, rotations


def distances_to(centers, rotations, index):
    rel = np.einsum('ji,njk->nik', rotations[index], rotations)
    tr = np.trace(rel, axis1=1, axis2=2)
    # Right-multiplication by Rz(pi) swaps the two fingers; same physical grasp.
    tr_swapped = -rel[:, 0, 0]-rel[:, 1, 1]+rel[:, 2, 2]
    angle = np.arccos(np.clip((np.maximum(tr, tr_swapped)-1)/2, -1, 1))
    pos = np.linalg.norm(centers-centers[index], axis=1)
    return np.sqrt((pos/SETTINGS['position_scale_m'])**2+
                   (angle/np.deg2rad(SETTINGS['rotation_scale_degrees']))**2)


def farthest_selection(poses, scores, count=40):
    centers, rotations = pose_components(poses)
    selected = [int(np.argmax(scores))]
    nearest = distances_to(centers, rotations, selected[0])
    for _ in range(count-1):
        nearest[selected] = -np.inf
        idx = int(np.argmax(nearest))
        if nearest[idx] <= 1e-5:
            raise ValueError('Not enough distinct poses after gripper symmetry')
        selected.append(idx)
        nearest = np.minimum(nearest, distances_to(centers, rotations, idx))
    return np.asarray(selected)


def coverage(poses, selected):
    centers, rotations = pose_components(poses)
    nearest = np.full(len(poses), np.inf)
    for idx in selected:
        nearest = np.minimum(nearest, distances_to(centers, rotations, idx))
    pair_min = []
    for idx in selected:
        d = distances_to(centers, rotations, idx)[selected]
        d[np.where(selected == idx)[0]] = np.inf
        pair_min.append(float(d.min()))
    return dict(nearest_reference_distance_mean=float(nearest.mean()),
        nearest_reference_distance_p95=float(np.percentile(nearest, 95)),
        nearest_reference_distance_max=float(nearest.max()),
        minimum_reference_separation=float(min(pair_min)))


def screen(mesh, hand, poses, scores, out, fingerprints):
    cache = out/'geometry-screen.npz'
    manifest = out/'geometry-screen.json'
    cache_key = dict(settings=SETTINGS, sources=fingerprints)
    if cache.exists() and manifest.exists():
        record = json.loads(manifest.read_text())
        if record['inputs'] != cache_key:
            raise ValueError('Geometry cache inputs changed; use a fresh output directory')
        a = np.load(cache)
        return a['valid'], a['plane_clearance_m'], record['counts']
    object_manager = CollisionManager(); object_manager.add_object('target', mesh)
    hand_manager = CollisionManager(); hand_manager.add_object('hand', hand)
    # A convex hull gives the exact minimum vertex height, without testing every vertex.
    hv = hand.convex_hull.vertices
    z = np.einsum('nj,vj->nv', poses[:, 2, :3], hv) + poses[:, 2, 3, None]
    clearance = z.min(axis=1)-mesh.bounds[0, 2]
    pre_clearance = clearance - SETTINGS['pregrasp_distance_m']*poses[:, 2, 2]
    score_ok = scores >= SETTINGS['score_threshold']
    table_ok = np.minimum(clearance, pre_clearance) >= SETTINGS['table_clearance_m']
    passed = score_ok & table_ok
    counts = dict(total=len(poses), score_pass=int(score_ok.sum()),
        score_and_plane_pass=int(passed.sum()), final_object_collision_rejected=0,
        approach_object_collision_rejected=0)
    offsets = np.linspace(0, SETTINGS['pregrasp_distance_m'],
        round(SETTINGS['pregrasp_distance_m']/SETTINGS['approach_sample_step_m'])+1)
    start = time.perf_counter()
    for n, i in enumerate(np.flatnonzero(passed)):
        for step, offset in enumerate(offsets):
            t = poses[i].copy(); t[:3, 3] -= offset*t[:3, 2]
            hand_manager.set_transform('hand', t)
            if object_manager.in_collision_other(hand_manager):
                passed[i] = False
                key = 'final_object_collision_rejected' if step == 0 else 'approach_object_collision_rejected'
                counts[key] += 1
                break
        if n % 500 == 0:
            print(out.name, 'geometry', n, 'elapsed', round(time.perf_counter()-start, 1), flush=True)
    counts['eligible'] = int(passed.sum())
    np.savez_compressed(cache, valid=passed, plane_clearance_m=clearance)
    manifest.write_text(json.dumps(dict(inputs=cache_key, counts=counts), indent=2)+'\n')
    return passed, clearance, counts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/graspgen-ycb-40refs')
    args = parser.parse_args()
    verify_bowl_conversion()
    hand = hand_geometry()
    report = dict(status='geometry_screened_not_physics_validated', settings=SETTINGS,
        frame='mesh_T_gripper is original GraspGen frame; position/quaternion are GARMI hand goals; quaternion wxyz; meters',
        graspgen_T_garmi_hand=HAND_OFFSET.tolist(), bowl_conversion_verified_count=40,
        coverage_scope='Finite generated candidates with score >= 0.5 passing geometry screening; not all possible grasps.',
        limitations=['No closure/contact dynamics or lift validation', 'No full GARMI IK or arm/scene collision checks',
            'Object approach checks are discrete at 5 mm spacing, not continuous collision detection',
            'No kitchen position mask; do not substitute for validated bowl task bank'], objects={})
    bank = dict(status=report['status'], frame=report['frame'], objects=[])
    for ycb, name in OBJECTS.items():
        out = args.out/ycb
        pool_path = out/'candidate_pool.npz'
        pool = np.load(pool_path)
        poses, scores = pool['T_object_graspgen'], pool['scores']
        assert np.isfinite(poses).all() and np.isfinite(scores).all()
        mesh_path = ROOT/'assets/ycb'/ycb/'google_16k/textured.obj'
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        fingerprints = dict(pool_sha256=sha(pool_path), mesh_sha256=sha(mesh_path),
            hand_sha256=sha(ROOT/'assets/graspgen/garmi-hand.stl'),
            urdf_sha256=sha(ROOT/'rl/assets/garmi_training.urdf'))
        valid, clearance, counts = screen(mesh, hand, poses, scores, out, fingerprints)
        indices = np.flatnonzero(valid)
        if len(indices) < 40:
            raise RuntimeError(f'{name}: only {len(indices)} eligible candidates')
        chosen_local = farthest_selection(poses[indices], scores[indices])
        chosen = indices[chosen_local]
        baseline = np.argsort(-scores[indices], kind='stable')[:40]
        stats = dict(sources=fingerprints, counts=counts,
            diverse_40=coverage(poses[indices], chosen_local),
            score_top40_baseline=coverage(poses[indices], baseline),
            selected_score_min=float(scores[chosen].min()), selected_score_max=float(scores[chosen].max()),
            selected_score_mean=float(scores[chosen].mean()), candidate_indices=chosen.tolist())
        ranks = np.empty(len(scores), dtype=int); ranks[np.argsort(-scores, kind='stable')] = np.arange(1, len(scores)+1)
        refs = []
        for j, i in enumerate(chosen):
            t = poses[i] @ HAND_OFFSET
            q = Rotation.from_matrix(t[:3, :3]).as_quat()[[3,0,1,2]]
            if q[0] < 0:
                q *= -1
            refs.append(dict(id=f'{name}-ref{j+1:03d}', rank=int(ranks[i]), score=float(scores[i]),
                mesh_T_gripper=poses[i].tolist(), position=t[:3,3].tolist(), quaternion=q.tolist()))
        target = ROOT/'resources'/f'{name}-40-references.json'
        target.write_text(json.dumps(refs, indent=2)+'\n')
        (out/'selected-40.json').write_text(json.dumps(refs, indent=2)+'\n')
        np.savez_compressed(out/'selected-40.npz', T_object_graspgen=poses[chosen],
            T_object_garmi_hand=poses[chosen]@HAND_OFFSET, scores=scores[chosen],
            candidate_indices=chosen, plane_clearance_m=clearance[chosen])
        stats['reference_file'] = str(target.relative_to(ROOT)); stats['reference_sha256'] = sha(target)
        report['objects'][name] = stats
        bank['objects'].append(dict(name=name, ycb_name=ycb, mesh=str(mesh_path.relative_to(ROOT)),
            mesh_sha256=fingerprints['mesh_sha256'], bounds=mesh.bounds.tolist(), references=refs))
        print(name, json.dumps(stats, indent=2), flush=True)
        (args.out/'selection-report.json').write_text(json.dumps(report, indent=2)+'\n')
    (ROOT/'resources/ycb-40-reference-selection.json').write_text(json.dumps(report, indent=2)+'\n')
    (ROOT/'assets/grasp_bank/ycb-selected-40.json').write_text(json.dumps(bank, indent=2)+'\n')


if __name__ == '__main__':
    main()
