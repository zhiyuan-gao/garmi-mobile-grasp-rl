"""Compare the same 40 pitcher references against scan and cooked collision surfaces."""
import argparse,json,pickle
from pathlib import Path
import numpy as np,trimesh,manifold3d as md
from rebalance_mug import FingerContacts
from select_references import ROOT,HAND_OFFSET,sha
from prepare_rl_validation import hand_parts
from trimesh.collision import CollisionManager

ASSET=ROOT/'assets/ycb/019_pitcher_base'


def cooked_boundary(path):
    parts=[p.convex_hull for _,p in pickle.load(path.open('rb'))]
    union=md.Manifold.batch_boolean([md.Manifold(md.Mesh64(np.asarray(p.vertices),np.asarray(p.faces,dtype=np.uint64))) for p in parts],md.OpType.Add)
    assert union.status()==md.Error.NoError
    output=union.to_mesh64()
    mesh=trimesh.Trimesh(output.vert_properties[:,:3],output.tri_verts,process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent
    return mesh


def audit(candidate,out):
    out.mkdir(parents=True,exist_ok=True)
    refs=json.loads((ROOT/'runs/ycb-rl-validation/019_pitcher_base/selected-40.json').read_text())
    surface=trimesh.load(ASSET/'google_16k/textured.obj',force='mesh',process=False)
    meshes={'scan':surface,'original':cooked_boundary(ASSET/'cooked.pkl'),'candidate':cooked_boundary(candidate/'cooked.pkl')}
    checkers={name:FingerContacts(mesh) for name,mesh in meshes.items()}
    hand=trimesh.util.concatenate(hand_parts());hand.apply_transform(HAND_OFFSET)
    manager=CollisionManager();manager.add_object('hand',hand)
    obstacles={name:CollisionManager() for name in meshes}
    for name,m in meshes.items():obstacles[name].add_object('target',m)
    d=np.array([.071,.069,0.]);d/=np.linalg.norm(d);split=np.array([-.015,.03,0.])+d*.068
    records=[]
    for ref in refs:
        pose=np.array(ref['mesh_T_gripper']);contacts={name:c.query(pose) for name,c in checkers.items()}
        row=dict(reference=ref['id'],contacts=contacts,approach_collision={},comparison={})
        for name,obstacle in obstacles.items():
            hits=[]
            for offset in np.linspace(0,.08,17):
                t=pose.copy();t[:3,3]-=offset*t[:3,2];manager.set_transform('hand',t)
                if manager.in_collision_other(obstacle):hits.append(float(offset))
            row['approach_collision'][name]=hits
        row['both_contacts_on_outer_handle']=all(c is not None and (np.array(c['point_m'])-split)@d>0 for c in contacts['scan'])
        for name in ['original','candidate']:
            delta=[None if a is None or b is None else float(b['q_m']-a['q_m']) for a,b in zip(contacts['scan'],contacts[name])]
            missing=any((a is None)!=(b is None) for a,b in zip(contacts['scan'],contacts[name]))
            row['comparison'][name]=dict(finger_q_delta_m=delta,missing_contact_disagreement=missing,
                flagged=missing or any(x is not None and abs(x)>.002 for x in delta))
        records.append(row)
    report=dict(candidate=str(candidate.relative_to(ROOT)),reference_count=len(refs),
        threshold_m=.002,threshold_scope='Engineering screening threshold for per-finger first-contact travel difference; not real-world accuracy certification.',
        method='Independent finger closure in 1 mm steps with binary refinement; cooked convex vertices re-hulled and boolean-unioned to remove internal faces.',
        limitations=['Static first contact is not force closure or grasp success.',
            'Median contact point can move along a contact patch; joint-travel difference is the primary metric.',
            'Reconstructed boundary from cooked vertices is an approximation of PhysX contact geometry.',
            'Pitcher body remains solid; no interior reconstruction.'],
        sources=dict(scan=sha(ASSET/'google_16k/textured.obj'),original_cooked=sha(ASSET/'cooked.pkl'),candidate_cooked=sha(candidate/'cooked.pkl'),
            references=sha(ROOT/'runs/ycb-rl-validation/019_pitcher_base/selected-40.json')),summary={},references=records)
    for name in ['original','candidate']:
        values=[abs(v) for r in records for v in r['comparison'][name]['finger_q_delta_m'] if v is not None]
        report['summary'][name]=dict(flagged_references=[r['reference'] for r in records if r['comparison'][name]['flagged']],
            max_finger_travel_difference_m=max(values),p95_finger_travel_difference_m=float(np.percentile(values,95)),
            approach_collision_references=[r['reference'] for r in records if r['approach_collision'][name]],
            two_contact_references=sum(all(c is not None for c in r['contacts'][name]) for r in records))
    report['outer_handle_references']=sum(r['both_contacts_on_outer_handle'] for r in records)
    (out/'contact-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['candidate','summary','outer_handle_references']},indent=2),flush=True)
    return report,meshes


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--candidate',type=Path,default=ASSET/'collision-v2')
    p.add_argument('--out',type=Path,default=ROOT/'runs/pitcher-handle-audit/before');a=p.parse_args()
    audit(a.candidate.resolve(),a.out.resolve())
