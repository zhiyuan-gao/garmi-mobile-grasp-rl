"""Build local, reindexed expert inputs for retained grasps; do not run a planner."""
import argparse
import copy
import json

import numpy as np
import yaml

from prepare_grasp_repeatability import OUT, SOURCES, ASSETS
from prepare_rl_validation import save
from select_references import ROOT, sha


def inside(path):return '/workspace/'+str(path.relative_to(ROOT))


def prepare(name):
    folder=OUT/name
    review=json.loads((folder/'review.json').read_text())
    refs=json.loads((folder/'retained-references.json').read_text())
    if not refs:
        raise ValueError(f'{name}: no retained grasps; cannot create an executable expert configuration')
    out=folder/'expert-inputs';out.mkdir(exist_ok=False)
    bank=json.loads((SOURCES[name]/'bank.json').read_text());obj=bank['objects'][0]
    obj['references']=refs
    meta=json.loads((ASSETS[name]/'cooked.json').read_text())
    assert obj['usd_sha256']==meta['usd_sha256']==sha(ASSETS[name]/'rl.usda')
    assert meta['cooked_sha256']==sha(ASSETS[name]/'cooked.pkl')
    obj.update(cooked=inside(ASSETS[name]/'cooked.pkl'),cooked_sha256=sha(ASSETS[name]/'cooked.pkl'))
    bank['status']='local_repeatability_passed_expert_approach_pending_user_confirmation'
    save(out/'bank.json',bank)
    accepted=[c for c in review['conditions'] if c['passed']]
    halfwidth=float(np.abs(review['criteria']['object_offsets_m']).max())
    save(out/'endpoint-conditions.json',dict(conditions=accepted,
        note='Tested endpoint configurations, not expert initial joint states. Planner does not enforce these base/joint targets; its actual reached endpoint must be checked anew.'))
    ids=[r['id'] for r in refs]
    grouped={}
    for c in accepted:
        xy=tuple(np.round([.3-c['object_state'][1],6.35+c['object_state'][0]],6))
        grouped.setdefault(xy,[]).append(c)
    regions=[];cells=[];settings_regions=[]
    for i,(xy,conditions) in enumerate(sorted(grouped.items())):
        bounds=[(np.array(xy)-halfwidth).tolist(),(np.array(xy)+halfwidth).tolist()]
        name_region=f"{conditions[0]['region']}-{i:02d}"
        regions.append(dict(name=name_region,bounds_map_xy=bounds,shape_xy=[1,1],cell_offset=i))
        cells.append(dict(region=name_region,bounds_map_xy=bounds,
            allowed_reference_indices=sorted({ids.index(c['reference']) for c in conditions}),
            accepted_condition_ids=[c['id'] for c in conditions]))
        settings_regions.append(dict(name=name_region,bounds_map_xy=bounds,probability=1./len(grouped)))
    # Different original 2 cm cells produce non-overlapping 1 cm neighborhoods.
    for i,a in enumerate(regions):
        for b in regions[i+1:]:
            overlap=np.minimum(a['bounds_map_xy'][1],b['bounds_map_xy'][1])-np.maximum(a['bounds_map_xy'][0],b['bounds_map_xy'][0])
            assert not (overlap>1.e-8).all()
    oldmask=json.loads((SOURCES[name]/'position-mask.json').read_text())
    mask=copy.deepcopy(oldmask)
    mask.update(bank_path=str((out/'bank.json').relative_to(ROOT)),reference_ids=ids,
        regions=regions,cells=cells,cell_size_m=2*halfwidth,empty_cells=[],status='validated',
        allowed_cells_per_reference=[sum(i in c['allowed_reference_indices'] for c in cells) for i in range(len(refs))],
        method=f'Reindexed retained references; non-overlapping {2*halfwidth:g} m local windows inside original geometry-checked cells, with all 8 object perturbation points passed.',
        scope='Geometry-safe local object windows with finite physical evidence. Continuous physical success and full approach trajectories are not certified.',
        placement_adjustment='Only accepted local windows; no sampling across the original whole-counter rectangles.')
    mask['sources_sha256'].pop(oldmask['bank_path'])
    for path in [out/'bank.json',folder/'review.json',ASSETS[name]/'cooked.pkl',out/'endpoint-conditions.json']:
        mask['sources_sha256'][str(path.relative_to(ROOT))]=sha(path)
    # Retained subcells must be completely within a former cell allowing that ref.
    for cell in cells:
        lo,hi=np.array(cell['bounds_map_xy'])
        for idx in cell['allowed_reference_indices']:
            previous_idx=oldmask['reference_ids'].index(ids[idx])
            assert any((lo>=np.array(old['bounds_map_xy'])[0]-1.e-8).all() and
                (hi<=np.array(old['bounds_map_xy'])[1]+1.e-8).all() and previous_idx in old['allowed_reference_indices'] for old in oldmask['cells'])
    save(out/'position-mask.json',mask)
    cfg=yaml.safe_load((SOURCES[name]/'task.yaml').read_text());task=cfg['garmi_task']
    task['multi_object']['manifest']=inside(out/'bank.json')
    task['multi_object']['reference_ids']={name:ids}
    task['kitchen'].update(mask=inside(out/'position-mask.json'),bowl_regions=settings_regions,sampling_mode='validated_subregions')
    task['kitchen']['files_sha256']['mask']=sha(out/'position-mask.json')
    # Initial states only. Original parked start remains distinct from the tested
    # terminal base/joint configuration; the untested approach will be judged later.
    audit=[];evaluation=[]
    for c in accepted:
        for k,(dx,dy) in enumerate([(0.,0.),(halfwidth/2,0.),(0.,halfwidth/2)]):
            state=c['object_state'].copy();state[0]+=dx;state[1]+=dy
            row=dict(id=f"audit-{c['id']}-{k}",object=name,reference=c['reference'],
                object_state=state,robot_state=[0.,.05,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                region=c['region'],condition_id=c['id'])
            audit.append(row)
        state=c['object_state'].copy();state[0]-=halfwidth/2;state[1]-=halfwidth/2
        evaluation.append(dict(id=f"eval-{c['id']}",object=name,reference=c['reference'],
            object_state=state,robot_state=[0.,.05,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],region=c['region']))
    save(out/'audit-cases.json',audit);save(out/'evaluation-cases.json',evaluation)
    assert not {tuple(c['robot_state']+c['object_state']) for c in audit}&{tuple(c['robot_state']+c['object_state']) for c in evaluation}
    cfg['evaluation']=dict(cases=inside(out/'evaluation-cases.json'),cases_sha256=sha(out/'evaluation-cases.json'))
    cfg['expert_preflight']=dict(status='awaiting_user_confirmation',
        endpoint_conditions=inside(out/'endpoint-conditions.json'),audit_cases=inside(out/'audit-cases.json'),
        endpoint_constraints_enforced_by_planner=False,
        note='Prepared initial states only. Full approach and actual reached terminal configurations have not been tested.')
    (out/'task.yaml').write_text(yaml.safe_dump(cfg))
    save(out/'preparation.json',dict(status='inputs_prepared_pending_runtime_reset_check_and_user_confirmation',
        object=name,reference_count=len(refs),accepted_conditions=len(accepted),local_window_count=len(cells),
        object_window_halfwidth_m=halfwidth,
        audit_cases=len(audit),evaluation_cases=len(evaluation),
        default_configuration_changed=False,expert_trajectories_generated=0,optimizer_updates=0,
        asset_sha256={key:obj[key+'_sha256'] for key in ('mesh','usd','cooked')},
        limitations=['Local windows are supported by finite probes, not exhaustive continuous physics validation.',
            'Expert motion is still untested and planner may reach a different base/arm endpoint.',
            'No random initial-base generalization claimed; initial audit base is the existing parked pose.']))
    print(name,len(refs),'refs',len(cells),'local windows',len(audit),'audit initial states',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('object',choices=SOURCES);args=p.parse_args()
    prepare(args.object)
