"""One bounded supplemental pitcher batch, then repeatability of any successes."""
import argparse
import copy
import json
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from select_references import ROOT,HAND_OFFSET,pose_components,distances_to,sha
from prepare_rl_validation import save

OUT=ROOT/'runs/pitcher-supplement-v3'
SOURCE=ROOT/'runs/pitcher-fresh-v3/019_pitcher_base'
TARGET=OUT/'019_pitcher_base'


def prepare():
    import prepare_fresh_pitcher as pipeline
    OUT.mkdir(exist_ok=False);TARGET.mkdir()
    pool=np.load(SOURCE/'candidate_pool.npz');poses,scores=pool['T_object_graspgen'],pool['scores']
    records=json.loads((SOURCE/'candidate-contacts.json').read_text())['records']
    old=set(np.load(SOURCE/'selected-40.npz')['candidate_indices'].tolist())
    byindex={r['candidate_index']:r for r in records}
    eligible=np.array([r['candidate_index'] for r in records if r['eligible'] and r['candidate_index'] not in old])
    centers,rotations=pose_components(poses[eligible]);chosen=[]
    nearest=np.full(len(eligible),np.inf)
    for local in np.argsort(-scores[eligible],kind='stable'):
        if nearest[local]<.5:continue
        chosen.append(int(eligible[local]));nearest=np.minimum(nearest,distances_to(centers,rotations,local))
        if len(chosen)==40:break
    assert len(chosen)==40 and not set(chosen)&old
    for file in ['candidate_pool.npz','geometry-screen.npz','geometry-screen.json','cooked-boundary.obj']:
        (TARGET/file).symlink_to(os.path.relpath(SOURCE/file,TARGET))
    ranks=np.empty(len(scores),int);ranks[np.argsort(-scores,kind='stable')]=np.arange(1,len(scores)+1)
    refs=[]
    for j,i in enumerate(chosen):
        t=poses[i]@HAND_OFFSET;q=Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]
        if q[0]<0:q*=-1
        refs.append(dict(id=f'pitcher-v3b-ref{j+1:03d}',rank=int(ranks[i]),score=float(scores[i]),
            mesh_T_gripper=poses[i].tolist(),position=t[:3,3].tolist(),quaternion=q.tolist()))
    save(TARGET/'selected-40.json',refs)
    np.savez_compressed(TARGET/'selected-40.npz',T_object_graspgen=poses[chosen],
        T_object_garmi_hand=poses[chosen]@HAND_OFFSET,scores=scores[chosen],candidate_indices=chosen)
    pipeline.OUT=TARGET;pipeline.BOUNDARY=TARGET/'cooked-boundary.obj'
    save(TARGET/'selection-report.json',dict(status='supplemental_geometry_screened_physics_pending',
        sources=pipeline.fingerprints(),source_pool_sha256=sha(SOURCE/'candidate_pool.npz'),
        method='Previously untested qualified candidates, descending model score with minimum pose distance 0.5; one fixed batch of 40.',
        score_min=float(scores[chosen].min()),score_max=float(scores[chosen].max()),
        references=[dict(reference=ref['id'],**byindex[i]) for ref,i in zip(refs,chosen)],
        limitations=['Model score is not calibrated physical success probability.','No change to repeatability acceptance criteria.']))
    try:
        pipeline.prepare()
    except AssertionError:
        mask_path=TARGET/'position-mask.json'
        if not mask_path.exists() or json.loads(mask_path.read_text())['status']!='failed':raise
        finish()
    print('SUPPLEMENT_READY',float(scores[chosen].min()),float(scores[chosen].max()),flush=True)


def finish():
    """Explicitly crop only empty exterior grid strips; reject interior holes."""
    import prepare_fresh_pitcher as pipeline
    pipeline.OUT=TARGET;pipeline.BOUNDARY=TARGET/'cooked-boundary.obj'
    mask=json.loads((TARGET/'position-mask.json').read_text())
    assert mask['status']=='failed'
    save(TARGET/'position-mask-before-supported-crop.json',mask)
    regions=[];cells=[];removed=[]
    for region in mask['regions']:
        old=[c for c in mask['cells'] if c['region']==region['name']]
        shape=np.array(region['shape_xy']);valid=np.array([bool(c['allowed_reference_indices']) for c in old]).reshape(shape)
        active=np.argwhere(valid);assert len(active)>0
        lo=active.min(0);hi=active.max(0)+1
        assert valid[lo[0]:hi[0],lo[1]:hi[1]].all(),'Interior mask holes must not be hidden'
        subset=[old[x*shape[1]+y] for x in range(lo[0],hi[0]) for y in range(lo[1],hi[1])]
        origin=np.array(region['bounds_map_xy'])[0]
        bounds=[(origin+lo*mask['cell_size_m']).tolist(),(origin+hi*mask['cell_size_m']).tolist()]
        regions.append(dict(region,bounds_map_xy=bounds,shape_xy=(hi-lo).tolist(),cell_offset=len(cells)))
        cells.extend(subset);removed.append(dict(region=region['name'],removed_cells=len(old)-len(subset),bounds_map_xy=bounds))
    mask.update(regions=regions,cells=cells,empty_cells=[],status='validated',
        allowed_cells_per_reference=[sum(i in c['allowed_reference_indices'] for c in cells) for i in range(40)],
        placement_adjustment='Explicitly cropped empty exterior strips for the supplemental reference set; full prior failed mask retained.',
        exterior_crop=removed)
    save(TARGET/'position-mask.json',mask)
    cfg=yaml.safe_load((SOURCE/'task.yaml').read_text());task=cfg['garmi_task']
    task['multi_object']['manifest']=pipeline.inside(TARGET/'bank.json')
    task['multi_object']['reference_ids']={'pitcher':mask['reference_ids']}
    task['kitchen']['mask']=pipeline.inside(TARGET/'position-mask.json')
    for r,v in zip(task['kitchen']['bowl_regions'],regions):r['bounds_map_xy']=v['bounds_map_xy']
    task['kitchen']['files_sha256']['mask']=sha(TARGET/'position-mask.json')
    (TARGET/'task.yaml').write_text(yaml.safe_dump(cfg))
    pipeline.audit();print('EXPLICIT_EXTERIOR_CROP',removed,flush=True)


def repeat_inputs():
    import export_successful_grasps as export
    export.SOURCES={'pitcher':TARGET};export.OUT=OUT/'success-evidence';export.main()
    import prepare_grasp_repeatability as repeat
    repeat.SOURCES={'pitcher':TARGET};repeat.OUT=OUT/'repeatability';repeat.EVIDENCE=OUT/'success-evidence';repeat.main()
    summary=json.loads((OUT/'success-evidence/summary.json').read_text())['objects']['pitcher']
    if summary['reference_count']==0:
        folder=OUT/'repeatability/pitcher'
        save(folder/'retained-references.json',[])
        save(folder/'review.json',dict(status='no_successful_seed_conditions_for_repeatability',object='pitcher',
            condition_count=0,passed_condition_count=0,tested_reference_count=0,retained_reference_count=0,
            retained_handle_reference_count=0,physical_case_count=0,physical_counts={},conditions=[],retained_reference_ids=[],
            criteria=repeat.CRITERIA,expert_trajectories_generated=0,optimizer_updates=0))
        print('NO_SUCCESSFUL_SEEDS_NO_REPEAT_SIMULATION',flush=True)


def review():
    import review_grasp_repeatability as module
    module.OUT=OUT/'repeatability';module.SOURCES={'pitcher':TARGET};module.review('pitcher')


def expert():
    import prepare_repeatability_expert as module
    module.OUT=OUT/'repeatability';module.SOURCES={'pitcher':TARGET};module.prepare('pitcher')


def narrow_inputs():
    report=json.loads((OUT/'repeatability/pitcher/review.json').read_text())
    # One prospectively defined smaller-domain check, only for repeatable nominal
    # conditions that already tolerated at least three base perturbations.
    chosen={c['id'] for c in report['conditions'] if c['successes']['nominal']==5
            and c['successes']['base_offset']>=3 and c['successes']['object_offset']>=6}
    assert chosen
    evidence=json.loads((OUT/'success-evidence/pitcher/success-evidence.json').read_text())
    keep=[]
    for row in evidence['references']:
        row=copy.deepcopy(row)
        row['successful_conditions']=[c for c in row['successful_conditions'] if
            f"{row['reference']}-{c['region']}-{c['depth']}" in chosen]
        if row['successful_conditions']:keep.append(row)
    evidence['references']=keep;evidence['reference_count']=len(keep)
    evidence['selection_scope']='Single smaller-domain experiment; no change to success-count thresholds. Original broad-domain failures retained.'
    save(OUT/'narrow-evidence/pitcher/success-evidence.json',evidence)
    import prepare_grasp_repeatability as repeat
    repeat.SOURCES={'pitcher':TARGET};repeat.OUT=OUT/'narrow-repeatability';repeat.EVIDENCE=OUT/'narrow-evidence'
    repeat.CRITERIA=copy.deepcopy(repeat.CRITERIA)
    repeat.CRITERIA['object_offsets_m']=(np.array(repeat.CRITERIA['object_offsets_m'])*.5).tolist()
    repeat.CRITERIA['interpretation']='Prospectively frozen +/-2.5 mm object domain, base +/-5 mm unchanged; all success-count thresholds unchanged. Prior +/-5 mm failures retained.'
    repeat.main()


def narrow_review():
    import review_grasp_repeatability as module
    module.OUT=OUT/'narrow-repeatability';module.SOURCES={'pitcher':TARGET};module.review('pitcher')


def narrow_expert():
    import prepare_repeatability_expert as module
    module.OUT=OUT/'narrow-repeatability';module.SOURCES={'pitcher':TARGET};module.prepare('pitcher')


if __name__=='__main__':
    steps={'prepare':prepare,'finish':finish,'repeat-inputs':repeat_inputs,'review':review,'expert':expert,
           'narrow-inputs':narrow_inputs,'narrow-review':narrow_review,'narrow-expert':narrow_expert}
    p=argparse.ArgumentParser(__doc__);p.add_argument('step',choices=steps)
    args=p.parse_args();steps[args.step]()
