"""Plan from ordinary parked-arm starts for locally validated wrist-policy grasps."""
import argparse
import copy
import numpy as np
import yaml
from prepare_rl_validation import save
from prepare_wrist_v6 import OUT
from select_references import ROOT,sha
from try_handle_roll import read
from wrist_v6_inputs import build


def prepare(name, label, mode, include_passed=False, handle_only=False, self_buffer_mm=None,distal_buffer_mm=None):
    status=read(OUT/'status.json')['objects'][name]
    rows=[]
    for ref, evidence in status['endpoint_evidence'].items():
        if ref in status['accepted_reference_ids'] and not include_passed:
            continue
        if handle_only and not evidence[0]['condition']['handle']:
            continue
        # Preserve every passing placement; alternate placements can need different routes.
        source=evidence[0]['source']
        bank=read(ROOT/source/'bank.json')['objects'][0]
        reference=next(r for r in bank['references'] if r['id']==ref)
        conditions=[copy.deepcopy(e['condition']) for e in evidence if e['source']==source]
        rows.append(dict(ref=reference,source=source,handle=conditions[0]['handle'],conditions=conditions))
    assert rows, 'No outstanding locally validated references'
    assert not (OUT/name/label).exists(), 'Use a new label; never overwrite frozen inputs'
    folder=build(name,label,rows)
    configure_cases(name,folder,mode,self_buffer_mm,distal_buffer_mm)


def configure_cases(name,folder,mode,self_buffer_mm=None,distal_buffer_mm=None):
    label=folder.name
    cfg=yaml.safe_load((folder/'task.yaml').read_text())
    if self_buffer_mm is not None or distal_buffer_mm is not None:
        cfg['expert_collision']=dict(model='usd_adjacent')
        if self_buffer_mm is not None:
            assert 1<=self_buffer_mm<=30
            cfg['expert_collision']['self_buffer_m']=self_buffer_mm/1000
        if distal_buffer_mm is not None:
            assert 1<=distal_buffer_mm<=50
            cfg['expert_collision']['distal_buffer_m']=distal_buffer_mm/1000
        (folder/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        inputs=read(folder/'inputs.json');inputs['configuration_sha256']=sha(folder/'task.yaml');save(folder/'inputs.json',inputs)
    map_lo,map_hi=np.asarray(cfg['garmi_task']['kitchen']['base_bounds_map_xy'])
    task_lo=np.array([map_lo[1]-6.35,.3-map_hi[0]])+.015
    task_hi=np.array([map_hi[1]-6.35,.3-map_lo[0]])-.015
    cases=[]
    for i,c in enumerate(read(folder/'conditions.json')):
        base=np.array([0.,.05,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        if mode=='near_endpoint':
            base[:2]=np.clip(c['base_pose'][:2],task_lo,task_hi)
        elif mode=='lateral':
            base[:2]=np.clip([-.05,c['base_pose'][1]],task_lo,task_hi)
        elif mode!='center':
            raise ValueError(mode)
        assert (base[:2]>=task_lo).all() and (base[:2]<=task_hi).all()
        cases.append(dict(id=f'{name}-{label}-{i:03d}',object=name,reference=c['reference'],
                          object_state=c['object_state'],robot_state=base.tolist()))
    save(folder/'approach-cases.json',cases)
    count=len({c['reference'] for c in cases})
    save(folder/'approach-design.json',dict(mode=mode,reference_count=count,case_count=len(cases),
        initial_arm='Standard PARK; no endpoint IK state restore or teleport',
        base_yaw_task_radians=0.,base_bounds_task_xy=[task_lo.tolist(),task_hi.tolist()],
        scope='Diagnostic route validation at selected allowed starts, not random-reset success-rate estimation'))
    print(folder,count,'references',len(cases),'cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('object',choices=['mug','pitcher']);p.add_argument('label')
    p.add_argument('--mode',choices=['center','near_endpoint','lateral'],default='near_endpoint')
    p.add_argument('--include-passed',action='store_true')
    p.add_argument('--handle-only',action='store_true')
    p.add_argument('--self-buffer-mm',type=float)
    p.add_argument('--distal-buffer-mm',type=float)
    a=p.parse_args();prepare(a.object,a.label,a.mode,a.include_passed,a.handle_only,a.self_buffer_mm,a.distal_buffer_mm)
