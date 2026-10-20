"""Archive observed successful grasps and their conditions, without expert trajectories.

This is evidence for subsequent repeatability checks, not a production grasp bank.
Never promote all geometrically allowed placement cells based on one success.
"""
import json
from pathlib import Path

from select_references import ROOT, sha

SOURCES = {
    'mug': ROOT/'runs/mug-surface-v3/025_mug',
    'pitcher': ROOT/'runs/pitcher-fresh-v3/019_pitcher_base',
}
OUT = ROOT/'runs/grasp-success-review-v3'


def main():
    OUT.mkdir(exist_ok=False)
    overview = {}
    for name, source in SOURCES.items():
        physics = json.loads((source/'physics/summary.json').read_text())
        cases = json.loads((source/'physics/cases.json').read_text())
        bank = json.loads((source/'bank.json').read_text())
        selection = json.loads((source/'selection-report.json').read_text())
        assert physics['status'] == 'completed'
        assert not physics.get('diagnostic_overrides'), 'Diagnostic parameter overrides need an explicit task profile and fresh validation before promotion'
        assert physics['configuration_sha256'] == sha(source/'task.yaml')
        obj = bank['objects'][0]
        for key in ('mesh', 'usd'):
            assert sha(ROOT/obj[key].removeprefix('/workspace/')) == obj[key+'_sha256']
        def key(row):
            return row['reference'], row['region'], row['depth']
        by_case = {key(c): c for c in cases}
        assert len(by_case) == len(cases) == physics['case_count']
        selected = {r['reference']: r for r in selection['references']}
        refs = [r for r in obj['references'] if physics['per_reference'][r['id']]['successes'] > 0]
        assert len(refs) == physics['references_with_success']
        evidence = []
        for ref in refs:
            outcomes = [r for r in physics['results'] if r['reference'] == ref['id']]
            success = []
            for row in outcomes:
                if not row['passed']:
                    continue
                case = by_case[key(row)]
                success.append(dict(region=row['region'], depth=row['depth'],
                    object_state=case['object_state'],
                    tested_robot_base_pose=row['base_pose'],
                    tested_right_arm_joint_positions=row['arm_position'],
                    diagnostics=row))
            evidence.append(dict(reference=ref['id'], handle=selected[ref['id']]['handle'],
                status='observed_success_pending_repeatability_and_approach_validation',
                summary=physics['per_reference'][ref['id']], successful_conditions=success,
                all_diagnostic_outcomes=outcomes))
        target = OUT/name
        target.mkdir()
        (target/'references.json').write_text(json.dumps(refs, indent=2)+'\n')
        record = dict(status='observed_success_only_not_expert_ready', object=name,
            geometry={k:v for k,v in obj.items() if k != 'references'},
            source_directory=str(source.relative_to(ROOT)),
            sources_sha256={str((source/path).relative_to(ROOT)):sha(source/path) for path in (
                'bank.json','position-mask.json','task.yaml','selected-40.json',
                'physics/summary.json','physics/cases.json','selection-report.json')},
            reference_count=len(refs), references=evidence,
            minimum_followup=['Repeat successful conditions and test nearby placements.',
                'Bind each retained reference to its empirically supported placement/base conditions.',
                'Prepare a matching bank, reindexed position mask and cooked collision assets for planning.',
                'Obtain user confirmation before any expert approach trajectory generation.'],
            limitations=['Four original attempts are four placements, not repeated trials at the same condition.',
                'Saved base/joint values are direct IK initialization conditions, not a motion plan.',
                'No successful region has been inferred from a single successful point.',
                'References are not certified repeatable or reachable by a collision-free trajectory.'],
            expert_trajectories_generated=0, optimizer_updates=0)
        (target/'success-evidence.json').write_text(json.dumps(record, indent=2)+'\n')
        overview[name] = dict(reference_count=len(refs),
            successful_cases=sum(r['summary']['successes'] for r in evidence),
            handle_reference_count=sum(r['handle'] for r in evidence),
            references_at_least_three_of_four=[r['reference'] for r in evidence if r['summary']['successes']>=3],
            references_all_four=[r['reference'] for r in evidence if r['summary']['successes']==4],
            reference_file=str((target/'references.json').relative_to(ROOT)),
            evidence_file=str((target/'success-evidence.json').relative_to(ROOT)))
    (OUT/'summary.json').write_text(json.dumps(dict(status='repeatability_review_pending',objects=overview,
        default_configuration_changed=False,expert_trajectories_generated=0,optimizer_updates=0), indent=2)+'\n')
    print(json.dumps(overview, indent=2))


if __name__ == '__main__':
    main()
