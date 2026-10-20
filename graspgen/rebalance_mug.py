"""Select ten handle and thirty other two-contact mug references, without training."""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from trimesh.collision import CollisionManager
from scipy.spatial.transform import Rotation
from select_references import ROOT, HAND_OFFSET, origin, farthest_selection, sha


class FingerContacts:
    def __init__(self, mesh):
        self.object = CollisionManager()
        self.object.add_object('object', mesh)
        self.fingers = []
        urdf = ET.parse(ROOT/'rl/assets/garmi_training.urdf').getroot()
        for name in ('right_fr3_leftfinger', 'right_fr3_rightfinger'):
            link = urdf.find(f"link[@name='{name}']")
            joint = next(j for j in urdf.findall('joint') if j.find('child').get('link') == name)
            parts = []
            for c in link.findall('collision'):
                part = trimesh.creation.box(np.fromstring(c.find('geometry/box').get('size'), sep=' '))
                part.apply_transform(origin(c.find('origin')))
                parts.append(part)
            manager = CollisionManager()
            manager.add_object('finger', trimesh.util.concatenate(parts))
            self.fingers.append((manager, origin(joint.find('origin')),
                                 np.fromstring(joint.find('axis').get('xyz'), sep=' ')))

    def query(self, pose, opening_m=.08):
        assert 0 < opening_m <= .08
        maximum = opening_m/2
        result = []
        for manager, joint, axis in self.fingers:
            def collision(q):
                shift = np.eye(4)
                shift[:3, 3] = axis*q
                manager.set_transform('finger', pose @ HAND_OFFSET @ joint @ shift)
                return self.object.in_collision_other(manager, return_data=True)
            contact = None
            for q in np.linspace(maximum, 0, int(np.ceil(maximum/.001))+1):
                hit, data = collision(q)
                if hit:
                    lo, hi = q, min(q+.001, maximum)
                    for _ in range(7):
                        middle = (lo+hi)/2
                        if collision(middle)[0]:
                            lo = middle
                        else:
                            hi = middle
                    _, data = collision(lo)
                    contact = dict(q_m=float(lo), point_m=np.median([d.point for d in data], axis=0).tolist())
                    break
            result.append(contact)
        return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'runs/ycb-rl-validation')
    p.add_argument('--source', type=Path, default=ROOT/'runs/graspgen-ycb-40refs/025_mug')
    p.add_argument('--mesh', type=Path, default=ROOT/'assets/ycb/025_mug/google_16k/textured.obj')
    p.add_argument('--id-prefix', default='mug-ref')
    args = p.parse_args()
    out = args.out/'025_mug'
    out.mkdir(parents=True, exist_ok=True)
    source = args.source
    pool = np.load(source/'candidate_pool.npz')
    screen = np.load(source/'geometry-screen.npz')
    poses, scores = pool['T_object_graspgen'], pool['scores']
    mesh_path = args.mesh
    cache = out/'candidate-contacts.json'
    fingerprints = dict(pool=sha(source/'candidate_pool.npz'), mesh=sha(mesh_path),
                        urdf=sha(ROOT/'rl/assets/garmi_training.urdf'))
    if cache.exists():
        data = json.loads(cache.read_text())
        assert data['sources'] == fingerprints
        records = data['records']
    else:
        checker = FingerContacts(trimesh.load(mesh_path, force='mesh', process=False))
        records = []
        for n, i in enumerate(np.flatnonzero(screen['valid'])):
            contact = checker.query(poses[i])
            both = all(c is not None for c in contact)
            handle = both and all(c['point_m'][0] > .030 for c in contact)
            records.append(dict(candidate_index=int(i), contacts=contact,
                                both_contacts=both, handle=handle))
            if n % 200 == 0:
                print('CONTACT_SCREEN', n, 'handle', sum(r['handle'] for r in records), flush=True)
        cache.write_text(json.dumps(dict(sources=fingerprints, records=records), indent=2)+'\n')
    # The RL wrist is larger than the URDF palm used in the original screen.
    # Include its actual cooked collider before filling the handle quota.
    from prepare_rl_validation import hand_parts
    vertices=np.vstack([part.vertices for part in hand_parts()])
    height=trimesh.load(mesh_path,force='mesh',process=False).bounds[0,2]
    for r in records:
        t=poses[r['candidate_index']]@HAND_OFFSET
        z=(vertices@t[:3,:3].T+t[:3,3])[:,2].min()
        r['cooked_wrist_table_clear']=bool(min(z,z-.08*t[2,2])-height>=.002)
    handle = np.array([r['candidate_index'] for r in records if r['handle'] and r['cooked_wrist_table_clear']])
    other = np.array([r['candidate_index'] for r in records if r['both_contacts'] and not r['handle'] and r['cooked_wrist_table_clear']])
    if len(handle) < 10 or len(other) < 30:
        raise ValueError(f'Insufficient candidates: handle={len(handle)}, other={len(other)}')
    selected = np.concatenate([handle[farthest_selection(poses[handle], scores[handle], 10)],
                               other[farthest_selection(poses[other], scores[other], 30)]])
    ranks = np.empty(len(scores), dtype=int)
    ranks[np.argsort(-scores, kind='stable')] = np.arange(1, len(scores)+1)
    refs = []
    for j, i in enumerate(selected):
        t = poses[i] @ HAND_OFFSET
        q = Rotation.from_matrix(t[:3, :3]).as_quat()[[3, 0, 1, 2]]
        if q[0] < 0: q *= -1
        refs.append(dict(id=f'{args.id_prefix}{j+1:03d}', rank=int(ranks[i]), score=float(scores[i]),
            mesh_T_gripper=poses[i].tolist(), position=t[:3, 3].tolist(), quaternion=q.tolist()))
    (out/'selected-40.json').write_text(json.dumps(refs, indent=2)+'\n')
    np.savez_compressed(out/'selected-40.npz', T_object_graspgen=poses[selected],
        T_object_garmi_hand=poses[selected]@HAND_OFFSET, scores=scores[selected],
        candidate_indices=selected, plane_clearance_m=screen['plane_clearance_m'][selected])
    report = dict(status='geometry_only_pending_rl_validation', handle_count=10, total=40,
        eligible_handle_count=len(handle), eligible_other_two_contact_count=len(other),
        method='Independent finger first-contact geometry; both median contact X > 30 mm = handle. Farthest selection within 10/30 strata.',
        limitations='Independent closure is not force closure, synchronous closure, or lift success.',
        sources=fingerprints, selected_candidate_indices=selected.tolist(),
        references=[dict(reference=refs[j]['id'], **next(r for r in records if r['candidate_index']==i))
                    for j, i in enumerate(selected)])
    (out/'selection-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('MUG_REBALANCED', len(handle), len(other), out, flush=True)


if __name__ == '__main__':
    main()
