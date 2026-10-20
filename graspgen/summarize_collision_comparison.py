"""Summarize bounded collision diagnostics without promoting assets or references."""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/ycb-collision-v2'
report=dict(status='bounded_attempt_complete_candidates_not_promoted',expert_trajectories_generated=0,
    optimizer_updates=0,objects={},limitations=[
        'Direct IK endpoint initialization; no approach trajectory verification.',
        'Static endpoint precheck uses the unchanged scanned mesh, not either physical proxy.',
        'Mug wall thickness 1.5 mm is a modeling assumption; pitcher interior remains solid.',
        'Finite seeds and placements; counts are not general grasp success probabilities.',
        'Mass and friction fixed; PhysX recomputes inertia and center of mass for changed shapes.'])
rows=[]
for y in ['025_mug','019_pitcher_base']:
    a=json.loads((ROOT/'runs/ycb-rl-validation'/y/'physics/summary.json').read_text())
    b=json.loads((out/y/'physics/summary.json').read_text())
    assert a['case_count']==b['case_count']
    old_cases=json.loads((ROOT/'runs/ycb-rl-validation'/y/'physics/cases.json').read_text())
    new_cases=json.loads((out/y/'physics/cases.json').read_text())
    assert old_cases==new_cases
    item=dict(reference_count=40,cases=b['case_count'],case_poses_identical=True,
        original=dict(references_with_success=a['references_with_success'],counts=a['counts']),
        candidate=dict(references_with_success=b['references_with_success'],counts=b['counts']),
        gained_references=[],lost_references=[],missing_placement_regions=b['missing_placement_regions'])
    for ref,v in b['per_reference'].items():
        old=a['per_reference'][ref]
        rows.append(dict(object=y,reference=ref,attempts=v['attempts'],old_successes=old['successes'],new_successes=v['successes']))
        if v['successes'] and not old['successes']:item['gained_references'].append(ref)
        if old['successes'] and not v['successes']:item['lost_references'].append(ref)
    if y=='025_mug':
        refs=[f'mug-ref{i:03d}' for i in range(1,11)]
        item['handle_references_with_success']={k:sum(d['per_reference'][r]['successes']>0 for r in refs) for k,d in [('original',a),('candidate',b)]}
    report['objects'][y]=item
    construction=ROOT/'assets/ycb'/y/'collision-v2/construction.json'
    c=json.loads(construction.read_text());c['status']='geometry_and_rl_diagnostics_completed_not_promoted_expert_pending'
    construction.write_text(json.dumps(c,indent=2)+'\n')
(out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'reference-comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(report,indent=2))
