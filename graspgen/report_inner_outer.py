"""Audit and summarize the bounded inner/outer handle experiment."""
import json
from collections import Counter
from select_references import ROOT,sha
from prepare_rl_validation import save
from try_handle_roll import read
OUT=ROOT/'runs/ycb-handle-inner-outer-v2'
report=dict(status='completed',target='One finger in handle hole, other outside; object remains fixed, vertical Z is up.',
 scope='Direct IK endpoint initialization followed by physical settle/close/lift/hold; no expert approach trajectories or training.',
 default_banks_changed=False,experiments={},sources_sha256={})
for version,name in [(1,'mug'),(2,'pitcher'),(2,'mug')]:
 folder=ROOT/f'runs/ycb-handle-inner-outer-v{version}/{name}'
 summary=read(folder/'endpoint/summary.json');bank=read(folder/'bank.json');cases=read(folder/'placement-cases.json')
 assert summary['status']=='completed'
 assert summary['configuration_sha256']==sha(folder/'task.yaml')
 assert summary['placement_cases_sha256']==sha(folder/'placement-cases.json')
 assert summary['expert_trajectories_generated']==summary['optimizer_updates']==0
 mask=read(folder/'position-mask.json')
 for file,expected in mask['sources_sha256'].items():assert sha(ROOT/file)==expected,file
 good=[r for r in summary['results'] if r['passed']]
 for r in good:
  assert r['reason']=='success' and r['max_hold_s']>=2 and r['max_lift_m']>=.08 and r['max_forbidden_force_n']<=.1
  assert r['endpoint_geometry_clear'] and r['isaac_hand_error_m']<=.005 and r['isaac_hand_angle_error_rad']<=.05
 ids={r['reference'] for r in good}
 accepted=dict(status='endpoint_passed_only_pending_repeatability_and_full_approach',opening_m=summary['pregrasp_open_width_m'],
  references=[r for r in bank['objects'][0]['references'] if r['id'] in ids],
  conditions=[dict(result=r,initial_case=next(c for c in cases if c['id']==r['id'])) for r in good],
  source_summary_sha256=sha(folder/'endpoint/summary.json'),source_bank_sha256=sha(folder/'bank.json'))
 save(folder/'endpoint-passed-candidates.json',accepted)
 key=f'{name}-v{version}'
 report['experiments'][key]={k:summary[k] for k in ['reference_count','case_count','references_with_success','counts','pregrasp_open_width_m']}
 report['experiments'][key]['passed_reference_ids']=sorted(ids)
 for path in [folder/'task.yaml',folder/'bank.json',folder/'endpoint/summary.json',folder/'endpoint-passed-candidates.json']:
  report['sources_sha256'][str(path.relative_to(ROOT))]=sha(path)
for filename in ['graspgen/try_handle_inner_outer.py','graspgen/try_handle_inner_outer_oblique.py','graspgen/validate_rl.py','graspgen/rl_geometry.py',
                 'graspgen/prepare_rl_validation.py','graspgen/rebalance_mug.py','rl/garmi_rl/fixed_finish.py','rl/garmi_rl/config.py']:
 report['sources_sha256'][filename]=sha(ROOT/filename)
report['preflight']=dict(pitcher_direct_approach_candidates=2520,pitcher_direct_approach_geometry_passes=0,
 trial_openings_m=[.08,.04,.03,.02],default_open_width_m=.08,
 success_criteria='Original lift >=8 cm, hold >=2 s, two finger contacts, slip <2 cm, no forbidden contact >0.1 N, endpoint error <=5 mm/0.05 rad.',
 code_validation='Syntax checks; isolated action test covers approach/settle/open/close/reset for 80/40/30/20 mm; physical phase traces confirm narrow opening.')
save(OUT/'review.json',report)
print(json.dumps(report['experiments'],indent=2))
