"""Aggregate strict new-policy endpoint and full-approach evidence without filling quotas."""
from collections import defaultdict,Counter
import numpy as np
import yaml
from select_references import ROOT,sha
from prepare_rl_validation import save
from try_handle_roll import read
from prepare_wrist_v6 import OUT


def endpoint(folder):
 s=read(folder/'physics/summary.json');inputs=read(folder/'inputs.json');criteria=read(folder/'criteria.json')
 assert s['status']=='completed' and s['contact_policy']=='fingers_and_wrist'
 assert s['configuration_sha256']==inputs['configuration_sha256']
 assert s['prescribed_cases_sha256']==sha(folder/'cases.json')==inputs['cases_sha256']
 groups=defaultdict(list)
 for r in s['results']:groups[r['condition_id']].append(r)
 result=[]
 for condition in read(folder/'conditions.json'):
  rows=groups[condition['id']];kinds={kind:[r for r in rows if r['test_kind']==kind] for kind in ['nominal','object_offset','base_offset']}
  assert [len(v) for v in kinds.values()]==[5,8,4]
  counts={k:sum(r['passed'] for r in v) for k,v in kinds.items()}
  passed=counts['nominal']>=4 and counts['object_offset']==8 and counts['base_offset']>=3
  result.append(dict(**{k:v for k,v in condition.items() if k not in ['passed','successes','failures','result_case_ids']},passed=passed,successes=counts,
                     failures=dict(Counter(r['reason'] for r in rows if not r['passed'])),result_case_ids=[r['id'] for r in rows]))
 review=dict(status='completed',contact_policy=s['contact_policy'],conditions=result,criteria=criteria,
  passed_reference_ids=sorted({r['reference'] for r in result if r['passed']}),counts=s['counts'],
  sources_sha256={str((folder/f).relative_to(ROOT)):sha(folder/f) for f in ['physics/summary.json','inputs.json','cases.json','conditions.json','criteria.json']})
 save(folder/'review.json',review);return review


def strict_approach(row):
 return row['reason']=='success' and row['handover_trigger']==1 and row['handover_distance']<=.005 and row['handover_angle']<=.05 and row['close_start_distance']<=.005 and row['close_start_angle']<=.05 and row['height']>=.08 and row['contact'] and not row['supported'] and row['slip']<.02 and row['hold_seconds']>=2 and not row['collision_bodies']


def matching_condition(condition, row):
 """A successful approach must finish at a locally tested object placement."""
 nominal=np.asarray(condition['object_state']);actual=np.asarray(row['initial_case']['object_state'])
 return bool(np.max(np.abs(actual[:2]-nominal[:2]))<=.005+1e-7
             and abs(actual[2]-nominal[2])<1e-5
             and min(np.linalg.norm(actual[3:7]-nominal[3:7]),np.linalg.norm(actual[3:7]+nominal[3:7]))<1e-5)


def main():
 status=dict(status='validating',target_per_object=40,contact_policy='fingers_and_wrist',objects={})
 for name in ['pitcher','mug']:
  physical={};approach={};batches={}
  for folder in sorted((OUT/name).iterdir()):
   if not folder.is_dir():continue
   if (folder/'inputs.json').exists() and (folder/'physics/summary.json').exists():
    rev=endpoint(folder);batches[folder.name]=dict(local_passed=len(rev['passed_reference_ids']),physical_counts=rev['counts'])
    for c in rev['conditions']:
     if c['passed']:physical.setdefault(c['reference'],[]).append(dict(condition=c,folder=str(folder.relative_to(ROOT)),source=read(folder/'inputs.json')['source']))
   for p in folder.glob('approach*/summary.json'):
    if (p.parent/'excluded.json').exists():continue
    s=read(p)
    cfg=yaml.safe_load((folder/'task.yaml').read_text())
    assert s['configuration_sha256']==sha(folder/'task.yaml')
    assert cfg['garmi_task']['contact_policy']=='fingers_and_wrist'
    for row in s['results']:
     if strict_approach(row):
      trajectory=ROOT/row['file'].removeprefix('/workspace/')
      assert sha(trajectory)==row['sha256']
      approach.setdefault(row['reference'],[]).append(dict(result=row,summary=str(p.relative_to(ROOT))))
  accepted=sorted(ref for ref in set(physical)&set(approach)
                  if any(matching_condition(e['condition'],a['result']) for e in physical[ref] for a in approach[ref]))
  status['objects'][name]=dict(local_repeatability_passed=len(physical),strict_full_approach_passed=len(approach),both_passed=len(accepted),accepted_reference_ids=accepted,batches=batches,
    endpoint_evidence=physical,approach_evidence=approach)
 status['published_bundles']={}
 for name in ['pitcher','mug']:
  p=ROOT/f'resources/validated-ycb-v6/{name}/summary.json'
  if p.exists() and read(p)['status']=='complete':
   status['published_bundles'][name]=dict(reference_count=40,summary=str(p.relative_to(ROOT)),summary_sha256=sha(p))
 if len(status['published_bundles'])==2:status['status']='completed'
 save(OUT/'status.json',status)
 print({name:{k:v for k,v in s.items() if k in ['local_repeatability_passed','strict_full_approach_passed','both_passed','batches']} for name,s in status['objects'].items()})

if __name__=='__main__':main()
