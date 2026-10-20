"""Apply frozen local repeatability criteria; never generate expert trajectories."""
import argparse
import collections
import json

from prepare_grasp_repeatability import OUT, SOURCES, ASSETS
from prepare_rl_validation import save
from select_references import sha


def review(name):
    folder=OUT/name
    inputs=json.loads((folder/'inputs.json').read_text())
    criteria=json.loads((OUT/'criteria.json').read_text())
    assert inputs['criteria_sha256']==sha(OUT/'criteria.json')
    physics=json.loads((folder/'physics/summary.json').read_text())
    assert physics['status']=='completed'
    assert physics['configuration_sha256']==inputs['configuration_sha256']
    assert physics['prescribed_cases_sha256']==inputs['cases_sha256']==sha(folder/'cases.json')
    assert physics['additional_object_collision_mesh_sha256']==inputs['cooked_boundary_sha256']
    cases=json.loads((folder/'cases.json').read_text())
    assert {c['id'] for c in cases}=={r['id'] for r in physics['results']}
    assert len(physics['results'])==inputs['case_count']
    grouped=collections.defaultdict(list)
    for result in physics['results']:grouped[result['condition_id']].append(result)
    conditions=[]
    for c in json.loads((folder/'conditions.json').read_text()):
        results=grouped[c['id']]
        groups={kind:[r for r in results if r['test_kind']==kind] for kind in ('nominal','object_offset','base_offset')}
        assert [len(groups[k]) for k in groups]==[5,8,4]
        count={k:sum(r['passed'] for r in v) for k,v in groups.items()}
        passed=(count['nominal']>=criteria['nominal_minimum_successes'] and
                count['object_offset']>=criteria['object_minimum_successes'] and
                count['base_offset']>=criteria['base_minimum_successes'])
        conditions.append(dict(**c,passed=passed,successes=count,
            failures=dict(collections.Counter(r['reason'] for r in results if not r['passed'])),
            result_case_ids=[r['id'] for r in results]))
    passed=[c for c in conditions if c['passed']]
    identifiers={c['reference'] for c in passed}
    source_bank=json.loads((SOURCES[name]/'bank.json').read_text())
    refs=[r for r in source_bank['objects'][0]['references'] if r['id'] in identifiers]
    save(folder/'retained-references.json',refs)
    report=dict(status='local_repeatability_passed_expert_approach_pending' if refs else 'no_reference_passed_local_repeatability',
        object=name,criteria=criteria,condition_count=len(conditions),passed_condition_count=len(passed),
        tested_reference_count=inputs['reference_count'],retained_reference_count=len(refs),
        retained_handle_reference_count=len({c['reference'] for c in passed if c['handle']}),
        physical_case_count=physics['case_count'],physical_counts=physics['counts'],
        conditions=conditions,retained_reference_ids=[r['id'] for r in refs],
        sources_sha256=dict(inputs=sha(folder/'inputs.json'),physics=sha(folder/'physics/summary.json'),
            source_bank=sha(SOURCES[name]/'bank.json'),source_usd=sha(ASSETS[name]/'rl.usda')),
        limitations=['Finite deterministic simulation repeats, not an estimate of real-world reliability.',
            'Acceptance is local to listed object/base conditions; no whole-counter claim.',
            'Perturbations are tested individually, not all joint combinations.',
            'Expert approach and any resulting endpoint still require physical validation.'],
        expert_trajectories_generated=0,optimizer_updates=0)
    save(folder/'review.json',report)
    print(json.dumps({k:report[k] for k in ('object','status','condition_count','passed_condition_count',
        'tested_reference_count','retained_reference_count','retained_handle_reference_count','physical_counts')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('object',choices=SOURCES);args=p.parse_args()
    review(args.object)
