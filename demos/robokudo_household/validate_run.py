"""Validate saved evidence from one continuous perception-guided household run."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path


def validate(run):
    run=Path(run)
    sequence=json.loads((run/'sim/sequence.json').read_text())
    events=json.loads((run/'task/rl-transport-report.json').read_text())
    observations=json.loads((run/'sim/perception/index.json').read_text())
    by_id={v['id']:v for v in observations}
    assert set(by_id)=={'bowl-initial','mug-initial','mug-pregrasp'}, 'Missing perception observations'
    assert sequence['status']=='completed' and sequence['scene_initializations']==1
    assert sequence['mug_inside'] and abs(sequence['drawer_opening_m'])<.02
    assert sequence['bowl_shift']<.05 and sequence['mug_speed']<.02
    assert not any(e['event']=='failed' for e in events)
    transport=next(e for e in events if e['event']=='transport_result')
    assert transport['passed'], 'Bowl delivery failed'
    estimate=json.loads((run/'sim/perception/bowl-initial/estimate.json').read_text())
    station=next(e for e in events if e['event']=='pickup_station_selected')
    assert math.dist(station['object'],estimate['position_map'])<1e-5, 'Station selection ignored perceived bowl pose'
    assert next(e for e in events if e['event']=='rl_grasp_result')['result']['success']
    assert next(e for e in sequence['events'] if e['event']=='mug_rl_result')['result']=='success'
    fresh=datetime.fromisoformat(by_id['mug-pregrasp']['captured_at_utc']).timestamp()
    assert fresh>transport['time'], 'Mug grasp reused a pre-transport observation'
    counts={}
    with (run/'sim/policy-inputs.jsonl').open() as f:
        for line in f:
            label=json.loads(line)['observation_id'];counts[label]=counts.get(label,0)+1
    assert set(counts)=={'bowl-initial','mug-pregrasp'} and min(counts.values())>0
    for label in counts:
        isolation=json.loads((run/'sim/perception'/label/'input-isolation.json').read_text())
        assert all(isolation[k] for k in ('policy_works_without_simulated_object',
            'fixed_finish_uses_perceived_geometry','observation_changes_with_perceived_pose',
            'physical_scoring_uses_simulator'))
    return dict(status='passed',run=run.name,policy_steps=counts,observations=list(by_id),
        bowl_delivered=True,mug_inside=True,drawer_opening_m=sequence['drawer_opening_m'],
        bowl_shift_m=sequence['bowl_shift'],scene_initializations=sequence['scene_initializations'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    args=p.parse_args();print(json.dumps(validate(args.run),indent=2))
