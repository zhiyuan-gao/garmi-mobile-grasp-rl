"""Audit Giskard labels at actual learner states; expert actions are never PPO transitions."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from garmi_rl.giskard_io import COMMAND_VERSION
from garmi_rl.joint_delta import COMMAND_VERSION as DELTA_VERSION, JOINT_SCALE
from garmi_rl.expert_actions import PARK


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--queries',required=True)
    parser.add_argument('--expert-results',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    queries=json.loads(Path(args.queries).read_text())
    delta_flags = ['policy_last_action' in q['snapshot'] for q in queries]
    assert len(set(delta_flags)) == 1, 'Mixed action semantics in label requests'
    joint_delta = delta_flags[0]
    label_version = DELTA_VERSION if joint_delta else COMMAND_VERSION
    responses=json.loads(Path(args.expert_results).read_text())
    assert responses['status']=='complete'
    lookup={r['id']:r for r in responses['results']}
    assert len(lookup)==len(queries)==len({q['id'] for q in queries})
    rows=[]; states=[]; actions=[]; actual=[]; after=[]; terminals=[]
    rejected=collections.Counter()
    maximum_frame_error=0.
    for query in queries:
        response=lookup[query['id']]
        assert query['approach_stage'] in ('pregrasp','approach_grasp')
        if response['status']!='command_found':
            rejected[response['status']]+=1
            continue
        assert response['command_version']==COMMAND_VERSION
        x=np.asarray(query['snapshot']['observation'])
        y=np.asarray(response['action'])
        if joint_delta:
            measured = np.asarray(query['snapshot']['arm_position'])
            y = np.r_[y[:3], np.clip((PARK+1.5*y[3:]-measured)/JOINT_SCALE,-1.,1.)]
        learner=np.asarray(query['learner_action'])
        next_state=np.asarray(query['next_observation'])
        assert x.shape==next_state.shape==(64,) and y.shape==learner.shape==(10,)
        assert x[49]==next_state[49]==1.
        history_key = 'policy_last_action' if joint_delta else 'last_action'
        assert np.allclose(x[39:49],query['snapshot'][history_key],atol=1.e-6)
        assert np.allclose(next_state[39:49],learner,atol=1.e-6)
        diagnostic=response['diagnostic'];hand=np.array(diagnostic['hand_pose'])
        delta=hand[:3,:3].T @ (np.array(diagnostic['goal_position'])-hand[:3,3])
        error=float(np.max(np.abs(delta-x[30:33])));maximum_frame_error=max(maximum_frame_error,error)
        assert error<3.e-4
        if query['approach_stage']=='approach_grasp':
            assert diagnostic['stage']=='approach_grasp', 'Teacher phase restarted'
        rows.append(dict(id=query['id'],reference=query['reference'],initial_case=query['initial_case'],
            step=query['step'],terminal=query['terminal'],approach_stage=diagnostic['stage'],
            source='giskard_at_learner_state',expert_action_executed=False,successful_episode_required=False))
        states.append(x);actions.append(y);actual.append(learner);after.append(next_state);terminals.append(query['terminal'])
    assert rows,'No valid Giskard labels'
    arrays=dict(states=np.array(states,dtype=np.float32),expert_actions=np.array(actions,dtype=np.float32),
        learner_actions=np.array(actual,dtype=np.float32),next_states=np.array(after,dtype=np.float32),
        terminal=np.array(terminals,dtype=bool))
    assert all(np.isfinite(x).all() for x in arrays.values())
    target=out/'labels.npz';np.savez_compressed(target,**arrays)
    entry=dict(file=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        count=len(rows),samples=rows,command_version=label_version,
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('garmi_rl/giskard_expert.py','garmi_rl/giskard_io.py')})
    manifest=dict(status='passed',expert_backend='giskard',command_version=label_version,demonstrations=[],dagger_label_sets=[entry],
        labels=len(rows),rejected=dict(rejected),maximum_frame_error=maximum_frame_error,
        per_reference=dict(collections.Counter(r['reference'] for r in rows)),
        successful_learner_episodes_required=False,expert_actions_executed=False,
        physics_success_verified_for_all_labels=False,
        queries_sha256=hashlib.sha256(Path(args.queries).read_bytes()).hexdigest(),
        expert_results_sha256=hashlib.sha256(Path(args.expert_results).read_bytes()).hexdigest())
    (out/'index.json').write_text(json.dumps(manifest,indent=2))
    print('GISKARD_LABELS_DONE',len(rows),dict(rejected),flush=True)


if __name__=='__main__':
    main()
