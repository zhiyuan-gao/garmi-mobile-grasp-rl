"""Freeze repeatability cases and criteria before any simulation results exist."""
import copy
import itertools
import json
import random

import numpy as np

from audit_pitcher_contacts import cooked_boundary
from prepare_rl_validation import save
from select_references import ROOT, sha

OUT=ROOT/'runs/grasp-repeatability-v3'
EVIDENCE=ROOT/'runs/grasp-success-review-v3'
SOURCES={'mug':ROOT/'runs/mug-surface-v3/025_mug',
         'pitcher':ROOT/'runs/pitcher-fresh-v3/019_pitcher_base'}
ASSETS={'mug':ROOT/'assets/ycb/025_mug/surface-v3',
        'pitcher':ROOT/'assets/ycb/019_pitcher_base/contact-v3-coarse'}
CRITERIA=dict(nominal_attempts=5,nominal_minimum_successes=4,
    object_offsets_m=[list(x) for x in itertools.product([-.005,0.,.005],repeat=2) if any(x)],
    object_minimum_successes=8,base_offsets_m=[[.005,0.],[-.005,0.],[0.,.005],[0.,-.005]],
    base_minimum_successes=3,
    interpretation='Local engineering acceptance rule fixed before execution; not a statistical or real-world guarantee.')


def main():
    OUT.mkdir(exist_ok=False)
    save(OUT/'criteria.json',CRITERIA)
    for name,source in SOURCES.items():
        target=OUT/name;target.mkdir()
        evidence_path=EVIDENCE/name/'success-evidence.json'
        evidence=json.loads(evidence_path.read_text())
        for path,digest in evidence['sources_sha256'].items():assert sha(ROOT/path)==digest
        mask=json.loads((source/'position-mask.json').read_text())
        original_cases=json.loads((source/'physics/cases.json').read_text())
        originals={(c['reference'],c['region'],c['depth']):c for c in original_cases}
        cases=[];conditions=[]
        for ref in evidence['references']:
            for successful in ref['successful_conditions']:
                reference=ref['reference'];region=successful['region'];depth=successful['depth']
                condition=f'{reference}-{region}-{depth}'
                base=successful['tested_robot_base_pose'];arm=successful['tested_right_arm_joint_positions']
                original=originals[(reference,region,depth)]
                variants=[('nominal',[0.,0.])]*5+[
                    ('object_offset',x) for x in CRITERIA['object_offsets_m']]+[
                    ('base_offset',x) for x in CRITERIA['base_offsets_m']]
                for i,(kind,offset) in enumerate(variants):
                    case=copy.deepcopy(original)
                    case.update(id=f'{condition}-{i:02d}',condition_id=condition,test_kind=kind,
                        offset_task_xy_m=offset,endpoint_base_pose=base.copy(),endpoint_arm_seed=arm.copy())
                    if kind=='object_offset':
                        case['object_state'][:2]=(np.array(case['object_state'][:2])+offset).tolist()
                    if kind=='base_offset':
                        case['endpoint_base_pose'][:2]=(np.array(base[:2])+offset).tolist()
                    xy=np.array([.3-case['object_state'][1],6.35+case['object_state'][0]])
                    index=mask['reference_ids'].index(reference)
                    assert any((xy>=np.array(c['bounds_map_xy'])[0]).all() and
                        (xy<=np.array(c['bounds_map_xy'])[1]).all() and index in c['allowed_reference_indices'] for c in mask['cells'])
                    cases.append(case)
                conditions.append(dict(id=condition,reference=reference,region=region,depth=depth,
                    object_state=successful['object_state'],base_pose=base,arm_position=arm,handle=ref['handle']))
        # Spread identical nominal repetitions across batches and environment slots.
        random.Random(2026101901).shuffle(cases)
        save(target/'cases.json',cases);save(target/'conditions.json',conditions)
        cooked_boundary(ASSETS[name]/'cooked.pkl').export(target/'cooked-boundary.obj')
        save(target/'inputs.json',dict(source=str(source.relative_to(ROOT)),asset=str(ASSETS[name].relative_to(ROOT)),
            condition_count=len(conditions),case_count=len(cases),reference_count=len(evidence['references']),
            criteria_sha256=sha(OUT/'criteria.json'),cases_sha256=sha(target/'cases.json'),
            evidence_sha256=sha(evidence_path),cooked_sha256=sha(ASSETS[name]/'cooked.pkl'),
            cooked_boundary_sha256=sha(target/'cooked-boundary.obj'),configuration_sha256=sha(source/'task.yaml'),
            nominal_mode='Saved base and arm exactly; independent environment resets, deterministic simulation, not independent real-world samples.',
            perturbation_mode='Fixed prescribed base; local IK from saved arm seed. Object orientation, friction and mass unchanged.',
            expert_trajectories_generated=0,optimizer_updates=0))
        print(name,len(conditions),'conditions',len(cases),'cases',flush=True)


if __name__=='__main__':main()
