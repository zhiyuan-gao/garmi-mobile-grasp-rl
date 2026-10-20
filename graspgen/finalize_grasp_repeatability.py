"""Finalize saved local grasp subsets and matching pre-expert configurations."""
import json

from prepare_grasp_repeatability import OUT
from prepare_rl_validation import save
from select_references import ROOT,sha


def main():
    initial={name:json.loads((OUT/name/'review.json').read_text()) for name in ('mug','pitcher')}
    broad=ROOT/'runs/pitcher-supplement-v3/repeatability/pitcher'
    narrow=ROOT/'runs/pitcher-supplement-v3/narrow-repeatability/pitcher'
    additional=narrow if (narrow/'review.json').exists() else broad
    assert (additional/'review.json').exists(),'Supplemental review must finish before finalization'
    supplementary=json.loads((additional/'review.json').read_text())
    summaries={}
    for name,folder,review in [('mug',OUT/'mug',initial['mug']),('pitcher',additional,supplementary)]:
        expert=folder/'expert-inputs'
        ready=review['retained_reference_count']>0
        if ready:
            check=json.loads((expert/'reset-check/summary.json').read_text())
            assert check['status']=='passed' and check['configuration_sha256']==sha(expert/'task.yaml')
            assert check['expert_trajectories_generated']==check['optimizer_updates']==check['action_steps']==0
            preparation=json.loads((expert/'preparation.json').read_text())
            audit=json.loads((expert/'audit-cases.json').read_text())
            smoke=[];seen=set()
            for case in audit:
                if case['reference'] not in seen:
                    smoke.append(case);seen.add(case['reference'])
            assert len(smoke)==review['retained_reference_count']
            save(expert/'smoke-cases.json',smoke)
            preparation['initial_smoke_case_count']=len(smoke)
            preparation['smoke_cases_sha256']=sha(expert/'smoke-cases.json')
            preparation['status']='runtime_inputs_checked_awaiting_user_confirmation_for_expert_trajectories'
            preparation['reset_check_sha256']=sha(expert/'reset-check/summary.json')
            save(expert/'preparation.json',preparation)
        summaries[name]=dict(status=review['status'],retained_reference_count=review['retained_reference_count'],
            tested_object_offset_m=max(abs(x) for row in review['criteria']['object_offsets_m'] for x in row),
            retained_handle_reference_count=review['retained_handle_reference_count'],
            passed_condition_count=review['passed_condition_count'],
            references=str((folder/'retained-references.json').relative_to(ROOT)),
            review=str((folder/'review.json').relative_to(ROOT)),
            expert_configuration=str((expert/'task.yaml').relative_to(ROOT)) if ready else None,
            expert_initial_case_count=preparation['audit_cases'] if ready else 0,
            first_smoke_case_count=preparation['initial_smoke_case_count'] if ready else 0,
            runtime_inputs_checked=ready)
    preserved=json.loads((ROOT/'runs/pitcher-fresh-v3/preserved-inputs.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in preserved.items())
    total=initial['mug']['physical_case_count']+initial['pitcher']['physical_case_count']+supplementary['physical_case_count']
    if additional==narrow:total+=json.loads((broad/'review.json').read_text())['physical_case_count']
    supplemental_physics=json.loads((ROOT/'runs/pitcher-supplement-v3/019_pitcher_base/physics/summary.json').read_text())
    total+=supplemental_physics['case_count']
    report=dict(status='local_subsets_saved_expert_trajectories_pending_confirmation',objects=summaries,
        original_pitcher_references_retained=initial['pitcher']['retained_reference_count'],
        supplemental_pitcher_candidate_count=40,
        supplemental_pitcher_broad_domain_retained=json.loads((broad/'review.json').read_text())['retained_reference_count'],
        smaller_pitcher_domain_check_performed=additional==narrow,
        supplemental_pitcher_initial_physics= dict(case_count=supplemental_physics['case_count'],counts=supplemental_physics['counts']),
        total_physical_diagnostic_cases=total,criteria_sha256=sha(OUT/'criteria.json'),
        preserved_input_sha256=preserved,default_configuration_changed=False,
        expert_trajectories_generated=0,optimizer_updates=0,
        limitations=['Finite deterministic simulation checks and local object windows; not universal usable grasps.',
            'Object and base perturbations tested separately; simultaneous extremes not established.',
            'Expert planner does not constrain terminal base/joints to saved diagnostic values; full resulting motions and terminal grasps still need validation.',
            'No expert motion or training executed.'])
    save(OUT/'summary.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
