"""Save contact and physical comparison; preserve unresolved references explicitly."""
import json,csv
from pathlib import Path
from select_references import ROOT,sha
out=ROOT/'runs/pitcher-handle-audit';asset=ROOT/'assets/ycb/019_pitcher_base'
before=json.loads((out/'before/contact-audit.json').read_text())
primary=out/'coarse' if (out/'coarse/physics/summary.json').exists() else out/'after'
after=json.loads((primary/'contact-audit.json').read_text())
physics=json.loads((primary/'physics/summary.json').read_text())
baseline=json.loads((ROOT/'runs/ycb-collision-v2/019_pitcher_base/physics/summary.json').read_text())
original=json.loads((ROOT/'runs/ycb-rl-validation/019_pitcher_base/physics/summary.json').read_text())
assert physics['status']=='completed' and physics['configuration_sha256']==sha(primary/'task.yaml')
cases=json.loads((primary/'physics/cases.json').read_text())
assert cases==json.loads((ROOT/'runs/ycb-collision-v2/019_pitcher_base/physics/cases.json').read_text())
old_bank=json.loads((ROOT/'runs/ycb-rl-validation/019_pitcher_base/bank.json').read_text())['objects'][0]
new_bank=json.loads((primary/'bank.json').read_text())['objects'][0]
assert old_bank['references']==new_bank['references']
assert sha(asset/'google_16k/textured.obj')==old_bank['mesh_sha256']==after['sources']['scan']
assert sha(asset/'rl.usda')==old_bank['usd_sha256']
assert sha(asset/'collision-v2/rl.usda')==json.loads((asset/'collision-v2/cooked.json').read_text())['usd_sha256']
no_two=[r['reference'] for r in after['references'] if not all(c is not None for c in r['contacts']['candidate'])]
report=dict(status='bounded_collision_repair_and_physics_diagnostics_complete_residual_flags_retained',
    final_candidate=after['candidate'],final_test_configuration=str((primary/'task.yaml').relative_to(ROOT)),default_configuration_changed=False,
    original_mesh_and_pointcloud_source_unchanged=True,original_40_reference_poses_unchanged=True,
    previous_assets_preserved=True,case_poses_identical_to_collision_v2=True,
    before=before['summary']['candidate'],after=after['summary']['candidate'],
    physical_comparison={name:{k:d[k] for k in ['case_count','references_with_success','counts']} for name,d in [('original',original),('collision_v2',baseline),('corrected_final',physics)]},
    residual_contact_geometry_flags=after['summary']['candidate']['flagged_references'],
    references_without_two_static_contacts=no_two,placement_limitations=physics['missing_placement_regions'],
    reset=json.loads((primary/'physics/reset-check.json').read_text()),
    placement=json.loads((primary/'input-and-placement-audit.json').read_text()),
    expert_trajectories_generated=0,optimizer_updates=0,limitations=[
        'Repaired collision fit to existing scan; body still solid, no inner pitcher reconstruction.',
        'Four references retain >2 mm first-contact travel deviation and must remain flagged.',
        'Independent first-contact closure is not force closure or a physical grasp guarantee.',
        'Single finite diagnostic run; comparison does not establish a general success rate.',
        'No expert approach trajectory verification; requires user confirmation before generating trajectories.'])
if primary.name=='coarse':
    fine=json.loads((out/'after/physics/summary.json').read_text())
    report['physical_comparison']['fine_1mm_control']={k:fine[k] for k in ['case_count','references_with_success','counts']}
    assert cases==json.loads((out/'after/physics/cases.json').read_text())
    report['coarse_control_scope']='Same body construction; handle slabs changed from 1 mm to 4 mm. Both shape partition and approximation change; solver causality is not isolated.'
    report['physical_regression_remains']=physics['counts'].get('success',0)<baseline['counts'].get('success',0)
rows=[]
for a,b in zip(before['references'],after['references']):
    ref=b['reference'];d=b['comparison']['candidate'];p=physics['per_reference'][ref]
    rows.append(dict(reference=ref,old_flag=a['comparison']['candidate']['flagged'],new_flag=d['flagged'],
        left_travel_error_mm=None if d['finger_q_delta_m'][0] is None else d['finger_q_delta_m'][0]*1000,
        right_travel_error_mm=None if d['finger_q_delta_m'][1] is None else d['finger_q_delta_m'][1]*1000,
        two_static_contacts=ref not in no_two,old_physical_successes=baseline['per_reference'][ref]['successes'],
        new_physical_successes=p['successes'],physical_attempts=p['attempts']))
with (out/'reference-review.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
repair_path=ROOT/after['candidate']/'repair.json'
repair=json.loads(repair_path.read_text());repair['status']=report['status']
repair_path.write_text(json.dumps(repair,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['physical_comparison','residual_contact_geometry_flags','references_without_two_static_contacts']},indent=2))
