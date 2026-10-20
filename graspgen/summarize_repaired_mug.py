"""Record the complete repaired-surface experiment without publishing or training."""
import json
import numpy as np
from select_references import ROOT,sha,coverage
from prepare_repaired_mug import OUT,ASSET
physics=json.loads((OUT/'physics/summary.json').read_text())
assert physics['status']=='completed' and physics['configuration_sha256']==sha(OUT/'task.yaml')
refs=json.loads((OUT/'selected-40.json').read_text());selection=json.loads((OUT/'selection-report.json').read_text())
pool=np.load(OUT/'candidate_pool.npz');chosen=np.load(OUT/'selected-40.npz')['candidate_indices']
records=json.loads((OUT/'candidate-contacts.json').read_text())['records']
fingerprint=selection['sources'];assert fingerprint['pool']==sha(OUT/'candidate_pool.npz') and fingerprint['mesh']==sha(ASSET/'repaired.obj')
assert len(refs)==40 and selection['handle_count']==10
handle=[r['id'] for r in refs[:10]]
preserved={}
for y,name in [('025_mug','mug'),('019_pitcher_base','pitcher')]:
    old=json.loads((ROOT/'runs/ycb-rl-validation'/y/'bank.json').read_text())['objects'][0]
    preserved[name]=dict(original_usd_unchanged=sha(ROOT/'assets/ycb'/y/'rl.usda')==old['usd_sha256'],
        published_references_unchanged=json.loads((ROOT/'resources'/f'{name}-40-references.json').read_text())==old['references'])
    assert all(preserved[name].values())
report=dict(status='repaired_mesh_and_new_references_diagnosed_expert_pending_confirmation',
    repair=json.loads((ASSET/'repair.json').read_text()),
    candidate_count=len(pool['scores']),reference_count=40,handle_reference_count=10,
    eligible_handle_candidates=selection['eligible_handle_count'],eligible_other_candidates=selection['eligible_other_two_contact_count'],
    physical_case_count=physics['case_count'],references_with_success=physics['references_with_success'],physical_counts=physics['counts'],
    handle_references_with_success=sum(physics['per_reference'][r]['successes']>0 for r in handle),
    handle_success_cases=sum(physics['per_reference'][r]['successes'] for r in handle),
    reset=json.loads((OUT/'physics/reset-check.json').read_text()),
    input_and_placement_audit=json.loads((OUT.parent/'input-and-placement-audit.json').read_text()),
    published_assets_preserved=preserved,coverage={},expert_trajectories_generated=0,optimizer_updates=0,
    limitations=['Repaired approximate geometry, not a new measurement of the real YCB mug.',
        'New candidates and collision geometry changed together; results are not a controlled success-rate comparison to the original 40.',
        'Full-surface point cloud, not an occluded camera point cloud.',
        'Single finite diagnostic run per placement. No approach trajectory validation.',
        'No expert data or training; confirmation required before expert trajectory generation.'])
for kind,is_handle in [('handle',True),('other',False)]:
    eligible=np.array([r['candidate_index'] for r in records if r['both_contacts'] and r['handle']==is_handle])
    # Recompute the same wrist/table criterion used during selection.
    from prepare_rl_validation import hand_parts
    from select_references import HAND_OFFSET
    verts=np.vstack([p.vertices for p in hand_parts()]);bottom=np.array(report['repair']['bounds_m'])[0,2]
    valid=[]
    for i in eligible:
        t=pool['T_object_graspgen'][i]@HAND_OFFSET
        z=(verts@t[:3,:3].T+t[:3,3])[:,2].min()
        if min(z,z-.08*t[2,2])-bottom>=.002:valid.append(i)
    eligible=np.array(valid);picked=chosen[:10] if is_handle else chosen[10:]
    local=np.array([int(np.flatnonzero(eligible==i)[0]) for i in picked])
    baseline=np.argsort(-pool['scores'][eligible],kind='stable')[:len(picked)]
    report['coverage'][kind]=dict(diverse=coverage(pool['T_object_graspgen'][eligible],local),
        score_only_baseline=coverage(pool['T_object_graspgen'][eligible],baseline))
report['repair']['status']='continuous_surface_and_grasp_diagnostics_completed_expert_pending'
(ASSET/'repair.json').write_text(json.dumps(report['repair'],indent=2)+'\n')
selection['status']='rl_diagnostics_completed_expert_pending_confirmation'
(OUT/'selection-report.json').write_text(json.dumps(selection,indent=2)+'\n')
(OUT.parent/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['candidate_count','reference_count','eligible_handle_candidates','physical_case_count','references_with_success','physical_counts','handle_references_with_success','handle_success_cases']},indent=2))
