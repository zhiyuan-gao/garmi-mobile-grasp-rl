"""Expand around verified poses, reusing only endpoint seeds, never labels.

Every chosen pose is a previously untested screened GraspGen candidate. The
original physical criterion and full local-repeatability checks remain required.
"""
import copy
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from select_references import ROOT, HAND_OFFSET, sha, pose_components, distances_to
from prepare_rl_validation import save
from expand_reference_validation import prepare_geometry, TEMPLATES

OUT=ROOT/'runs/grasp-expansion-v4e'
TARGET=OUT/'pitcher'


def main(args):
    registry=ROOT/'runs/validated-reference-bank-v4/pitcher'
    seed_refs=json.loads((registry/'all-validated-references.json').read_text())
    seed_evidence=json.loads((registry/'validation-evidence.json').read_text())['references']
    assert len(seed_refs)>=2
    for e in seed_evidence:assert sha(ROOT/e['review'])==e['review_sha256']
    sources=[ROOT/'runs/pitcher-fresh-v3/019_pitcher_base',
             ROOT/'runs/pitcher-centered-pool-v4/019_pitcher_base',
             ROOT/'runs/pitcher-so3-pool-v4/019_pitcher_base']
    used=[set(),set(),set()]
    for folder in [sources[0],ROOT/'runs/pitcher-supplement-v3/019_pitcher_base',ROOT/'runs/grasp-expansion-v4a/pitcher']:
        used[0].update(np.load(folder/'selected-40.npz')['candidate_indices'].tolist())
    used[1].update(np.load(ROOT/'runs/grasp-expansion-v4c/pitcher/selected-40.npz')['candidate_indices'].tolist())
    vertical=np.load(ROOT/'runs/pitcher-vertical-pool-v4/019_pitcher_base/candidate_pool.npz')
    vertical_selection=np.load(ROOT/'runs/grasp-expansion-v4d/pitcher/selected-40.npz')['candidate_indices']
    used[2].update(vertical['parent_candidate_indices'][vertical_selection].tolist())
    for folder in args.exclude:
        for row in json.loads((folder/'pitcher/selection-report.json').read_text())['references']:
            used[row['parent_source']].add(row['parent_candidate_index'])
    poses=[];scores=[];records=[];parents=[]
    for j,source in enumerate(sources):
        pool=np.load(source/'candidate_pool.npz')
        contact=json.loads((source/'candidate-contacts.json').read_text())['records']
        parents.append(dict(directory=str(source.relative_to(ROOT)),pool_sha256=sha(source/'candidate_pool.npz'),
                            contacts_sha256=sha(source/'candidate-contacts.json')))
        for row in contact:
            i=row['candidate_index']
            if not row['eligible'] or i in used[j]:continue
            poses.append(pool['T_object_graspgen'][i]);scores.append(pool['scores'][i])
            records.append(dict(row,parent_source=j,parent_candidate_index=i))
    poses=np.array(poses);scores=np.array(scores)
    seed_poses=np.array([r['mesh_T_gripper'] for r in seed_refs])
    combined=np.concatenate([seed_poses,poses]);centers,rotations=pose_components(combined)
    distances=np.array([distances_to(centers,rotations,i)[len(seed_refs):] for i in range(len(seed_refs))])
    closest=distances.argmin(0);nearest=distances.min(0)
    chosen=[];assignments=[]
    # Round robin between successful families; nearest genuinely different pose
    # first. Enforce the requested separation from seeds and each new pose.
    while len(chosen)<args.count:
        before=len(chosen)
        for j in range(len(seed_refs)):
            eligible=(closest==j)&(distances[j]<=args.radius)&(nearest>=args.min_distance)
            if not eligible.any():continue
            i=int(np.argmin(np.where(eligible,distances[j],np.inf)))
            chosen.append(i);assignments.append(j)
            nearest=np.minimum(nearest,distances_to(centers,rotations,len(seed_refs)+i)[len(seed_refs):])
            nearest[i]=-np.inf
            if len(chosen)==args.count:break
        if len(chosen)==before:break
    assert len(chosen)>=40, f'Only {len(chosen)} distinct local candidates'
    TARGET.mkdir(parents=True,exist_ok=False)
    refs=[];selected_records=[]
    for k,(i,j) in enumerate(zip(chosen,assignments)):
        t=poses[i]@HAND_OFFSET;q=Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]
        if q[0]<0:q*=-1
        identifier=f'pitcher-{args.tag}-n{k+1:03d}'
        refs.append(dict(id=identifier,score=float(scores[i]),mesh_T_gripper=poses[i].tolist(),
                         position=t[:3,3].tolist(),quaternion=q.tolist()))
        selected_records.append(dict(records[i],reference=identifier,candidate_index=k,
            seed_reference=seed_refs[j]['id'],distance_to_seed=float(distances[j,i])))
    np.savez_compressed(TARGET/'selected-40.npz',T_object_graspgen=poses[chosen],
        T_object_garmi_hand=poses[chosen]@HAND_OFFSET,scores=scores[chosen],candidate_indices=np.arange(len(chosen)))
    save(TARGET/'selected-40.json',refs)
    save(TARGET/'selection-report.json',dict(status='untested_success_neighborhood_candidates',
        references=selected_records,parents=parents,seed_evidence=seed_evidence,
        minimum_pose_distance=args.min_distance,maximum_distance_to_seed=args.radius,
        method='Unchanged screened poses, excluding tested parent indices; round-robin local expansion around verified references. No success label is inherited.'))
    bank=json.loads((TEMPLATES['pitcher']/'bank.json').read_text());bank['objects'][0]['references']=refs
    save(TARGET/'bank.json',bank)
    prepare_geometry(SimpleNamespace(object='pitcher',out=OUT))
    mask=json.loads((TARGET/'position-mask.json').read_text());cases=[];missing=[]
    for k,(i,j) in enumerate(zip(chosen,assignments)):
        available=[]
        for c in seed_evidence[j]['conditions']:
            xy=np.array([.3-c['object_state'][1],6.35+c['object_state'][0]])
            if any((xy>=np.array(cell['bounds_map_xy'])[0]+.005-1e-7).all() and
                   (xy<=np.array(cell['bounds_map_xy'])[1]-.005+1e-7).all() and
                   k in cell['allowed_reference_indices'] for cell in mask['cells']):available.append(c)
        if not available:
            missing.append(refs[k]['id']);continue
        c=available[0]
        bases=[c['base_pose'].copy()]
        if args.two_bases:
            alternate=c['base_pose'].copy()
            # Moving forward to X=.32 collided with cabinets in the F batch.
            # Keep the validated forward coordinate and make a small lateral
            # move toward the target; full-body collision checks still apply.
            alternate[1]+=float(np.clip(c['object_state'][1]-alternate[1],-.03,.03))
            bases.append(alternate)
        for bi,base in enumerate(bases):
            cases.append(dict(id=f"{refs[k]['id']}-seed-{bi}",object='pitcher',reference=refs[k]['id'],
                region=c['region'],depth=f"seed-{j:02d}-{c['depth']}-base{bi}",object_state=c['object_state'],
                robot_state=[0.,.05,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                endpoint_base_pose=base,endpoint_arm_seed=c['arm_position'],test_kind='seed_search',
                seed_reference=seed_refs[j]['id']))
    assert len(cases)>=40
    save(TARGET/'seeded-cases.json',cases)
    save(TARGET/'seeded-case-audit.json',dict(case_count=len(cases),references_without_seed_placement=missing,
        cases_sha256=sha(TARGET/'seeded-cases.json'),
        method='Known successful base, optionally a second base moved laterally up to 3 cm toward the target; local IK from saved arm. Every new reference requires full validation.',
        expert_trajectories_generated=0,optimizer_updates=0))
    runner=(ROOT/'graspgen/run_expanded_validation.sh').read_text()
    needle='  --object-collision-mesh "/workspace/$batch/$name/cooked-boundary.obj"'
    assert runner.count(needle)==1
    runner=runner.replace(needle,needle+' --cases "/workspace/$batch/$name/seeded-cases.json"')
    (OUT/'runner.sh').write_text(runner)
    print('NEIGHBORHOOD_READY',len(refs),'candidates',len(cases),'seeded cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--tag',default='v4e')
    p.add_argument('--count',type=int,default=120)
    p.add_argument('--min-distance',type=float,default=.25)
    p.add_argument('--radius',type=float,default=1.)
    p.add_argument('--exclude',type=Path,action='append',default=[])
    p.add_argument('--two-bases',action='store_true')
    args=p.parse_args();OUT=args.out.resolve();TARGET=OUT/'pitcher'
    main(args)
