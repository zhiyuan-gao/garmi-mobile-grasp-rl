"""Fresh pitcher references: dual geometry/contact screening and isolated RL config.

Run sample_pool.py --fresh first, then this script's screen, select and prepare
steps. Sampling uses the unchanged original scan; physics uses contact-v3-coarse.
No expert trajectories or training are generated.
"""
import argparse
import itertools
import json
import pickle

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation
from trimesh.collision import CollisionManager

from audit_pitcher_contacts import cooked_boundary
from rebalance_mug import FingerContacts
from select_references import (ROOT, HAND_OFFSET, SETTINGS, sha, screen,
                               farthest_selection, coverage, verify_bowl_conversion)
from prepare_rl_validation import (hand_parts, build_mask,
                                    crop_to_supported_footprint, save, MASS_SOURCE)

OUT = ROOT/'runs/pitcher-fresh-v3/019_pitcher_base'
ASSET = ROOT/'assets/ycb/019_pitcher_base/contact-v3-coarse'
MESH = ROOT/'assets/ycb/019_pitcher_base/google_16k/textured.obj'
BOUNDARY = OUT/'cooked-boundary.obj'


def inside(path):
    return '/workspace/'+str(path.relative_to(ROOT))


def fingerprints():
    return dict(pool=sha(OUT/'candidate_pool.npz'), mesh=sha(MESH),
                cooked=sha(ASSET/'cooked.pkl'), usd=sha(ASSET/'rl.usda'),
                robot_cooked=sha(ROOT/'assets/kitchen/cooked-model.pkl'),
                urdf=sha(ROOT/'rl/assets/garmi_training.urdf'))


def screen_candidates():
    preserved_path=OUT.parent/'preserved-inputs.json'
    preserved={path:sha(ROOT/path) for path in [
        'configs/task.yaml','resources/pitcher-40-references.json','resources/mug-40-references.json',
        'runs/mug-surface-v3/validation.json','assets/ycb/019_pitcher_base/contact-v3-coarse/rl.usda']}
    if preserved_path.exists():
        assert json.loads(preserved_path.read_text())==preserved
    else:
        save(preserved_path,preserved)
    pool=np.load(OUT/'candidate_pool.npz')
    poses,scores=pool['T_object_graspgen'],pool['scores']
    assert np.isfinite(poses).all() and np.isfinite(scores).all()
    assert np.allclose(np.swapaxes(poses[:,:3,:3],1,2)@poses[:,:3,:3],np.eye(3),atol=1e-5)
    assert np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-5)
    provenance=json.loads((OUT.parent/'pool-provenance.json').read_text())['objects']['019_pitcher_base']
    assert provenance['original_count']==0 and provenance['mesh_sha256']==sha(MESH)
    assert provenance['pool_sha256']==sha(OUT/'candidate_pool.npz')
    cooked=cooked_boundary(ASSET/'cooked.pkl')
    # Diagnostic union boundary removes internal convex seams; never sampled as input.
    cooked.export(BOUNDARY)
    hand=trimesh.util.concatenate(hand_parts());hand.apply_transform(HAND_OFFSET)
    valid=np.ones(len(poses),bool);clearances=[];counts={}
    for label,mesh in [('scan',trimesh.load(MESH,force='mesh',process=False)),('cooked',cooked)]:
        target=OUT/label;target.mkdir(exist_ok=True)
        v,c,n=screen(mesh,hand,poses,scores,target,fingerprints())
        valid &= v;clearances.append(c);counts[label]=n
    np.savez_compressed(OUT/'geometry-screen.npz',valid=valid,plane_clearance_m=np.minimum(*clearances))
    save(OUT/'geometry-screen.json',dict(sources=fingerprints(),settings=SETTINGS,counts=counts,
        eligible_both=int(valid.sum()),cooked_boundary_sha256=sha(BOUNDARY)))
    print('DUAL_GEOMETRY_SCREEN',int(valid.sum()),counts,flush=True)


def select():
    pool=np.load(OUT/'candidate_pool.npz');poses,scores=pool['T_object_graspgen'],pool['scores']
    screened=np.load(OUT/'geometry-screen.npz')
    geometry=json.loads((OUT/'geometry-screen.json').read_text())
    assert geometry['sources']==fingerprints()
    assert geometry['cooked_boundary_sha256']==sha(BOUNDARY)
    cache=OUT/'candidate-contacts.json'
    if cache.exists():
        data=json.loads(cache.read_text());assert data['sources']==fingerprints()
        records=data['records']
    else:
        scan=FingerContacts(trimesh.load(MESH,force='mesh',process=False))
        cooked=FingerContacts(trimesh.load(BOUNDARY,force='mesh',process=False))
        direction=np.array([.071,.069,0.]);direction/=np.linalg.norm(direction)
        split=np.array([-.015,.03,0.])+direction*.068
        records=[]
        for n,i in enumerate(np.flatnonzero(screened['valid'])):
            a,b=scan.query(poses[i]),cooked.query(poses[i])
            both=all(c is not None for c in a+b)
            delta=[abs(x['q_m']-y['q_m']) for x,y in zip(a,b)] if both else []
            handle=both and all((np.array(c['point_m'])-split)@direction>0 for c in a+b)
            records.append(dict(candidate_index=int(i),scan_contacts=a,cooked_contacts=b,
                both_contacts=both,maximum_contact_travel_difference_m=max(delta) if delta else None,
                eligible=bool(both and max(delta)<=.002),handle=bool(handle)))
            if n%200==0:print('DUAL_CONTACT_SCREEN',n,'eligible',sum(r['eligible'] for r in records),flush=True)
        save(cache,dict(sources=fingerprints(),records=records))
    eligible=np.array([r['candidate_index'] for r in records if r['eligible']],dtype=int)
    assert len(eligible)>=40,f'Only {len(eligible)} eligible candidates'
    local=farthest_selection(poses[eligible],scores[eligible],40);selected=eligible[local]
    ranks=np.empty(len(scores),int);ranks[np.argsort(-scores,kind='stable')]=np.arange(1,len(scores)+1)
    refs=[];byindex={r['candidate_index']:r for r in records}
    for j,i in enumerate(selected):
        t=poses[i]@HAND_OFFSET;q=Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]
        if q[0]<0:q*=-1
        refs.append(dict(id=f'pitcher-v3-ref{j+1:03d}',rank=int(ranks[i]),score=float(scores[i]),
            mesh_T_gripper=poses[i].tolist(),position=t[:3,3].tolist(),quaternion=q.tolist()))
    assert not (OUT/'selected-40.json').exists()
    save(OUT/'selected-40.json',refs)
    np.savez_compressed(OUT/'selected-40.npz',T_object_graspgen=poses[selected],
        T_object_garmi_hand=poses[selected]@HAND_OFFSET,scores=scores[selected],candidate_indices=selected,
        plane_clearance_m=screened['plane_clearance_m'][selected])
    report=dict(status='geometry_screened_physics_pending',sources=fingerprints(),settings=SETTINGS,
        total_candidates=len(scores),geometry_eligible=int(screened['valid'].sum()),contact_eligible=len(eligible),
        reference_count=40,handle_reference_count=sum(byindex[i]['handle'] for i in selected),
        eligible_handle_count=sum(byindex[i]['handle'] for i in eligible),
        maximum_selected_contact_travel_difference_m=max(byindex[i]['maximum_contact_travel_difference_m'] for i in selected),
        contact_travel_threshold_m=.002,selected_candidate_indices=selected.tolist(),
        coverage=dict(diverse=coverage(poses[eligible],local),
            score_only=coverage(poses[eligible],np.argsort(-scores[eligible],kind='stable')[:40])),
        references=[dict(reference=ref['id'],**byindex[i]) for ref,i in zip(refs,selected)],
        method='Fresh candidates only; dual geometry and independent finger closure with <=2 mm travel disagreement; farthest-point pose selection, no handle quota.',
        limitations=['Coverage is within the finite screened pool, not all possible grasps.',
            'Independent first contacts do not establish synchronous contact, force closure or lift success.',
            'Cooked boundary reconstructed from convex vertices approximates PhysX contact geometry.'])
    save(OUT/'selection-report.json',report)
    print('SELECTED',json.dumps({k:report[k] for k in ['total_candidates','geometry_eligible','contact_eligible','handle_reference_count','coverage']}),flush=True)


def prepare():
    verify_bowl_conversion()
    refs=json.loads((OUT/'selected-40.json').read_text());assert len(refs)==40
    mesh=trimesh.load(MESH,force='mesh',process=False)
    bank_path=OUT/'bank.json';assert not bank_path.exists()
    save(bank_path,dict(status='fresh_pitcher_diagnostic_only_expert_pending',objects=[dict(name='pitcher',
        mesh=inside(MESH),mesh_sha256=sha(MESH),usd=inside(ASSET/'rl.usda'),usd_sha256=sha(ASSET/'rl.usda'),
        mass=.178,mass_source=MASS_SOURCE,friction_assumption=1.,bounds=mesh.bounds.tolist(),references=refs)]))
    mask=crop_to_supported_footprint(build_mask('pitcher',mesh,refs,bank_path),mesh)
    save(OUT/'position-mask.json',mask)
    assert mask['status']=='validated',mask['empty_cells']
    cfg=yaml.safe_load((ROOT/'configs/task.yaml').read_text());task=cfg['garmi_task']
    task['multi_object']=dict(manifest=inside(bank_path),objects=['pitcher'],split='all',reference_ids={'pitcher':[r['id'] for r in refs]})
    task['kitchen']['mask']=inside(OUT/'position-mask.json')
    for region,valid in zip(task['kitchen']['bowl_regions'],mask['regions']):region['bounds_map_xy']=valid['bounds_map_xy']
    task['kitchen']['files_sha256']=dict(mask=sha(OUT/'position-mask.json'),scene_usd=sha(ROOT/'assets/kitchen/kitchen.usda'))
    (OUT/'task.yaml').write_text(yaml.safe_dump(cfg))
    audit()


def audit():
    refs=json.loads((OUT/'selected-40.json').read_text());mask=json.loads((OUT/'position-mask.json').read_text())
    obj=json.loads((OUT/'bank.json').read_text())['objects'][0]
    for relative,digest in mask['sources_sha256'].items():assert sha(ROOT/relative)==digest
    for key in ['mesh','usd']:assert sha(ROOT/obj[key].removeprefix('/workspace/'))==obj[key+'_sha256']
    meta=json.loads((ASSET/'cooked.json').read_text())
    assert sha(ASSET/'rl.usda')==meta['usd_sha256'] and sha(ASSET/'cooked.pkl')==meta['cooked_sha256']
    selection=json.loads((OUT/'selection-report.json').read_text());assert selection['sources']==fingerprints()
    assert mask['reference_ids']==[r['id'] for r in refs]
    poses=np.load(OUT/'selected-40.npz')['T_object_graspgen']
    for ref,pose in zip(refs,poses):
        t=pose@HAND_OFFSET;q=np.array(ref['quaternion'])
        assert np.allclose(t[:3,:3],Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),atol=1e-6)
        assert np.allclose(t[:3,3],ref['position'],atol=1e-8)
    parts=[p.convex_hull for _,p in pickle.load((ASSET/'cooked.pkl').open('rb'))]
    bounds=trimesh.util.concatenate(parts).bounds
    scene=CollisionManager()
    for i,(r,m) in enumerate(pickle.load((ROOT/'assets/kitchen/scene_meshes.pkl').open('rb'))):scene.add_object(str(i),m.convex_hull if r['type']=='Mesh' else m)
    surfaces=[np.array(r['surface_bounds']) for r in json.loads((ROOT/'assets/reference_filter/proposal.json').read_text())['bowl_regions']]
    margins=[]
    for region,surface in zip(mask['regions'],surfaces):
        lo,hi=np.array(region['bounds_map_xy']);gaps=np.vstack([lo+bounds[0,:2]-surface[0,:2],surface[1,:2]-hi-bounds[1,:2]])
        margins.append(dict(region=region['name'],minimum_worktop_edge_gap_m=float(gaps.min())))
    assert min(m['minimum_worktop_edge_gap_m'] for m in margins)>=.01
    rz=np.array([[0,1,0],[-1,0,0],[0,0,1]])
    shifts=np.array(list(itertools.product([-.01,.01],[-.01,.01],[0.])))
    swept=[trimesh.convex.convex_hull(((p.vertices@rz.T)[:,None,:]+shifts).reshape(-1,3)) for p in parts]
    manager=CollisionManager();manager.add_object('swept',trimesh.util.concatenate(swept));collisions=[]
    for i,c in enumerate(mask['cells']):
        xy=np.mean(c['bounds_map_xy'],axis=0);t=np.eye(4);t[:3,3]=[xy[1]-6.35,.3-xy[0],mask['bowl_origin_z_m']]
        manager.set_transform('swept',t)
        if manager.in_collision_other(scene):collisions.append(i)
    assert not collisions
    missing=[refs[i]['id'] for i,c in enumerate(mask['allowed_cells_per_reference']) if not c]
    save(OUT.parent/'input-and-placement-audit.json',dict(status='passed',reference_frames_checked=40,
        hashes_verified=True,collision_pieces=len(parts),worktop_margins=margins,
        checked_placement_cells=len(mask['cells']),object_furniture_collision_cells=collisions,
        initial_bottom_clearance_m=float(mask['bowl_origin_z_m']+bounds[0,2]-.945),
        references_without_placement=missing,expert_trajectories_generated=0,optimizer_updates=0))
    print('PREPARED_AND_AUDITED','placement_counts',mask['allowed_cells_per_reference'],'missing',missing,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('step',choices=['screen','select','prepare','audit'])
    args=parser.parse_args()
    {'screen':screen_candidates,'select':select,'prepare':prepare,'audit':audit}[args.step]()
