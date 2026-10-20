"""Incremental candidate validation. No expert collection or training entry point.

Keep prior successes and test genuinely different, previously untested poses.
All batches retain their own source indices, hashes, failures and repeatability
criteria. A candidate is never counted as usable merely because it was selected.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

from select_references import ROOT, HAND_OFFSET, sha, pose_components, distances_to
from prepare_rl_validation import save, build_mask, crop_to_supported_footprint
from prepare_grasp_repeatability import ASSETS, CRITERIA

SOURCES = {'mug': ROOT/'runs/mug-surface-v3/025_mug',
           'pitcher': ROOT/'runs/pitcher-fresh-v3/019_pitcher_base'}
TEMPLATES = SOURCES.copy()
OLD = {'mug': [SOURCES['mug']], 'pitcher': [SOURCES['pitcher'],
       ROOT/'runs/pitcher-supplement-v3/019_pitcher_base']}


def inside(p):
    return '/workspace/'+str(p.relative_to(ROOT))


def crop_empty_exterior(mask):
    """Only discard explicitly recorded exterior strips, never interior holes."""
    if not mask['empty_cells']:
        return mask
    regions, cells, changes = [], [], []
    for region in mask['regions']:
        old=[c for c in mask['cells'] if c['region']==region['name']]
        shape=np.array(region['shape_xy'])
        valid=np.array([bool(c['allowed_reference_indices']) for c in old]).reshape(shape)
        active=np.argwhere(valid)
        assert len(active), region['name']
        lo,hi=active.min(0),active.max(0)+1
        assert valid[lo[0]:hi[0],lo[1]:hi[1]].all(), 'Interior holes require explicit region partitioning'
        subset=[old[x*shape[1]+y] for x in range(lo[0],hi[0]) for y in range(lo[1],hi[1])]
        origin=np.array(region['bounds_map_xy'])[0]
        bounds=[(origin+lo*mask['cell_size_m']).tolist(),(origin+hi*mask['cell_size_m']).tolist()]
        regions.append(dict(region,bounds_map_xy=bounds,shape_xy=(hi-lo).tolist(),cell_offset=len(cells)))
        cells.extend(subset)
        changes.append(dict(region=region['name'],removed_cells=len(old)-len(subset),bounds_map_xy=bounds))
    mask.update(regions=regions,cells=cells,empty_cells=[],status='validated',exterior_crop=changes,
        allowed_cells_per_reference=[sum(i in c['allowed_reference_indices'] for c in cells)
                                     for i in range(len(mask['reference_ids']))])
    return mask


def prepare(args):
    name=args.object;source=SOURCES[name];target=args.out/name
    target.mkdir(parents=True,exist_ok=False)
    pool=np.load(source/'candidate_pool.npz');poses,scores=pool['T_object_graspgen'],pool['scores']
    records=json.loads((source/'candidate-contacts.json').read_text())['records']
    records=[r for r in records if (r.get('eligible',r['both_contacts']))]
    old=OLD[name]+[Path(p) for p in args.exclude]
    used=set()
    for folder in old:
        used.update(np.load(folder/'selected-40.npz')['candidate_indices'].tolist())
    eligible=np.array([r['candidate_index'] for r in records if r['candidate_index'] not in used],dtype=int)
    byindex={r['candidate_index']:r for r in records}
    centers,rotations=pose_components(poses[eligible]);nearest=np.full(len(eligible),np.inf)
    # Alternate score and coverage. Symmetric fingers are accounted for by the
    # existing pose metric (3 cm translation / 30 degree rotation).
    chosen=[]
    handle_order=[i for i in np.argsort(-scores[eligible],kind='stable') if byindex[eligible[i]]['handle']]
    for iteration in range(args.count):
        valid=nearest>=args.min_distance
        if not valid.any():break
        if name=='mug' and handle_order:
            available=[i for i in handle_order if valid[i]]
            local=available[0] if available else None
            handle_order=available[1:]
        else:local=None
        if local is None:
            priority=scores[eligible] if iteration%2==0 else nearest
            local=int(np.argmax(np.where(valid,priority,-np.inf)))
        chosen.append(int(eligible[local]))
        nearest=np.minimum(nearest,distances_to(centers,rotations,local))
        nearest[local]=-np.inf
    assert len(chosen)==args.count, f'Only {len(chosen)} distinct poses available'
    refs=[]
    for i in chosen:
        t=poses[i]@HAND_OFFSET;q=Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]
        if q[0]<0:q*=-1
        refs.append(dict(id=f'{name}-{args.tag}-c{i:05d}',score=float(scores[i]),
            mesh_T_gripper=poses[i].tolist(),position=t[:3,3].tolist(),quaternion=q.tolist()))
    # Legacy filename retained for consumers; explicit count in manifests is authoritative.
    save(target/'selected-40.json',refs)
    np.savez_compressed(target/'selected-40.npz',T_object_graspgen=poses[chosen],
        T_object_garmi_hand=poses[chosen]@HAND_OFFSET,scores=scores[chosen],candidate_indices=chosen)
    for file in ['candidate_pool.npz','geometry-screen.npz','geometry-screen.json']:
        (target/file).symlink_to(os.path.relpath(source/file,target))
    save(target/'selection-report.json',dict(status='candidates_only_physics_pending',reference_count=len(refs),
        source_pool_sha256=sha(source/'candidate_pool.npz'),contacts_sha256=sha(source/'candidate-contacts.json'),
        excluded_directories=[str(x.relative_to(ROOT)) for x in old],minimum_pose_distance=args.min_distance,
        method='Alternate descending score and farthest pose; mug includes untested handle candidates first.',
        references=[dict(reference=r['id'],**byindex[i]) for r,i in zip(refs,chosen)]))
    bank=json.loads((TEMPLATES[name]/'bank.json').read_text());obj=bank['objects'][0];obj['references']=refs
    for key in ['mesh','usd']:
        assert sha(ROOT/obj[key].removeprefix('/workspace/'))==obj[key+'_sha256']
    save(target/'bank.json',bank)
    prepare_geometry(args)


def prepare_geometry(args):
    name=args.object;source=SOURCES[name];target=args.out/name
    assert not (target/'task.yaml').exists(), 'Use a fresh output directory for completed preparations'
    obj=json.loads((target/'bank.json').read_text())['objects'][0]
    refs=obj['references']
    selected=json.loads((target/'selection-report.json').read_text())['references']
    mesh=trimesh.load(ROOT/obj['mesh'].removeprefix('/workspace/'),force='mesh',process=False)
    mask=crop_to_supported_footprint(build_mask(name,mesh,refs,target/'bank.json'),mesh)
    if mask['empty_cells']:save(target/'mask-before-exterior-crop.json',mask)
    mask=crop_empty_exterior(mask)
    assert mask['status']=='validated'
    save(target/'position-mask.json',mask)
    cfg=yaml.safe_load((TEMPLATES[name]/'task.yaml').read_text());task=cfg['garmi_task']
    task['multi_object']['manifest']=inside(target/'bank.json')
    task['multi_object']['reference_ids']={name:[r['id'] for r in refs]}
    task['kitchen']['mask']=inside(target/'position-mask.json')
    for r,v in zip(task['kitchen']['bowl_regions'],mask['regions']):r['bounds_map_xy']=v['bounds_map_xy']
    task['kitchen']['files_sha256']['mask']=sha(target/'position-mask.json')
    (target/'task.yaml').write_text(yaml.safe_dump(cfg))
    from audit_pitcher_contacts import cooked_boundary
    cooked_boundary(ASSETS[name]/'cooked.pkl').export(target/'cooked-boundary.obj')
    save(target/'preparation.json',dict(status='candidate_physics_pending',reference_count=len(refs),
        handle_count=sum(r['handle'] for r in selected),asset_cooked_sha256=sha(ASSETS[name]/'cooked.pkl'),
        asset_usd_sha256=sha(ASSETS[name]/'rl.usda'),configuration_sha256=sha(target/'task.yaml'),
        references_without_placement=[refs[i]['id'] for i,n in enumerate(mask['allowed_cells_per_reference']) if n==0],
        expert_trajectories_generated=0,optimizer_updates=0))
    print('EXPANSION_READY',name,len(refs),flush=True)


def repeat(args):
    import export_successful_grasps as export
    export.SOURCES={args.object:args.out/args.object};export.OUT=args.out/f'evidence-{args.object}'
    export.main()
    evidence_out=args.out/f'repeat-seeds-{args.object}'
    evidence=json.loads((export.OUT/args.object/'success-evidence.json').read_text())
    for ref in evidence['references']:
        ranked=sorted(ref['successful_conditions'],key=lambda c:(-c['diagnostics']['max_hold_s'],
                      c['diagnostics']['isaac_hand_error_m']))
        ref['successful_conditions']=ranked[:args.max_conditions]
    evidence['selection_scope']='First repeatability pass: longest successful hold conditions per reference. All original successful and failed conditions remain in the separate source evidence archive.'
    save(evidence_out/args.object/'success-evidence.json',evidence)
    import prepare_grasp_repeatability as module
    module.SOURCES=export.SOURCES;module.EVIDENCE=evidence_out;module.OUT=args.out/f'repeat-{args.object}'
    module.CRITERIA=copy.deepcopy(CRITERIA)
    # Both objects start at the original +/-5 mm standard. Smaller domains are
    # not silently substituted in order to inflate accepted reference counts.
    module.main()


def review(args):
    import review_grasp_repeatability as module
    module.OUT=args.out/f'repeat-{args.object}';module.SOURCES={args.object:args.out/args.object}
    folder=module.OUT/args.object
    inputs=json.loads((folder/'inputs.json').read_text())
    if inputs['case_count']==0:
        original=json.loads((args.out/args.object/'physics/summary.json').read_text())
        assert original['status']=='completed' and original['references_with_success']==0
        save(folder/'retained-references.json',[])
        save(folder/'review.json',dict(status='no_initial_success_no_repeatability_seeds',
            object=args.object,retained_reference_count=0,retained_handle_reference_count=0,
            retained_reference_ids=[],conditions=[],initial_physics_counts=original['counts'],
            initial_physics_sha256=sha(args.out/args.object/'physics/summary.json'),
            expert_trajectories_generated=0,optimizer_updates=0))
        print('NO_VALIDATED_REFERENCES',args.object,flush=True)
        return
    module.review(args.object)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('step',choices=['prepare','prepare-geometry','repeat','review'])
    p.add_argument('--object',choices=SOURCES,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=160)
    p.add_argument('--tag',default='v4a')
    p.add_argument('--min-distance',type=float,default=.3)
    p.add_argument('--exclude',action='append',default=[])
    p.add_argument('--source',type=Path,help='Independent screened pool; old indices from a different pool are not excluded.')
    p.add_argument('--max-conditions',type=int,default=1)
    args=p.parse_args();args.out=args.out.resolve()
    if args.source:
        SOURCES[args.object]=args.source.resolve();OLD[args.object]=[]
    {'prepare':prepare,'prepare-geometry':prepare_geometry,'repeat':repeat,'review':review}[args.step](args)
