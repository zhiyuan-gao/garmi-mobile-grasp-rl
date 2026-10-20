"""Build a geometry-bound mixed-width bank from saved source windows."""
import copy
import numpy as np,yaml
from select_references import ROOT,sha
from prepare_rl_validation import save
from try_handle_roll import read,host,inside
from prepare_wrist_v6 import OUT,freeze_repeat


def build(name,label,rows):
 out=OUT/name/label;out.mkdir(parents=True,exist_ok=True)
 master=ROOT/f'resources/validated-ycb-v{4 if name=="mug" else 5}/{name}'
 obj=read(master/'bank.json')['objects'][0];refs=[];groups={};checked={};conditions=[]
 for row in rows:
  ref=copy.deepcopy(row['ref']);source=ROOT/row['source'];other=read(source/'bank.json')['objects'][0]
  assert other['mesh_sha256']==obj['mesh_sha256']
  ref['opening_m']=ref.get('opening_m',yaml.safe_load((source/'task.yaml').read_text())['garmi_task']['controller'].get('gripper_open_width_m',.08))
  index=len(refs);refs.append(ref)
  mask=read(source/'position-mask.json')
  for file,digest in mask['sources_sha256'].items():
   if file not in checked:checked[file]=sha(ROOT/file)
   assert checked[file]==digest,file
  for c in row['conditions']:
   c=copy.deepcopy(c);c['handle']=row['handle'];conditions.append(c)
   xy=np.array([.3-c['object_state'][1],6.35+c['object_state'][0]]);key=tuple(np.round(xy,6));lo=np.array(key)-.005;hi=np.array(key)+.005
   j=mask['reference_ids'].index(ref['id'])
   assert any((lo>=np.array(cell['bounds_map_xy'][0])-1e-7).all() and (hi<=np.array(cell['bounds_map_xy'][1])+1e-7).all() and j in cell['allowed_reference_indices'] for cell in mask['cells']),(ref['id'],key)
   groups.setdefault(key,set()).add(index)
 obj['references']=refs;save(out/'bank.json',dict(status='wrist_v6_physics_pending',objects=[obj]))
 regions=[];cells=[];settings=[]
 for i,(key,indices) in enumerate(sorted(groups.items())):
  region=f'local-{i:02d}';bounds=[(np.array(key)-.005).tolist(),(np.array(key)+.005).tolist()]
  regions.append(dict(name=region,bounds_map_xy=bounds,shape_xy=[1,1],cell_offset=i))
  cells.append(dict(region=region,bounds_map_xy=bounds,allowed_reference_indices=sorted(indices)))
  settings.append(dict(name=region,bounds_map_xy=bounds,probability=1/len(groups)))
 sources={str((out/'bank.json').relative_to(ROOT)):sha(out/'bank.json')}
 for row in rows:
  source=ROOT/row['source'];sources[str((source/'position-mask.json').relative_to(ROOT))]=sha(source/'position-mask.json')
 mask=read(master/'position-mask.json')
 mask.update(bank_path=str((out/'bank.json').relative_to(ROOT)),reference_ids=[r['id'] for r in refs],sources_sha256=sources,regions=regions,cells=cells,
  cell_size_m=.01,status='validated',empty_cells=[],allowed_cells_per_reference=[sum(i in cell['allowed_reference_indices'] for cell in cells) for i in range(len(refs))],
  method='Reindexed 1 cm local windows wholly covered by source geometry masks at the same per-reference opening.',
  scope='Geometry clearance only; all physical labels must be retested with wrist permission.')
 save(out/'position-mask.json',mask);save(out/'evaluation-cases.json',[])
 cfg=yaml.safe_load((master/'task.yaml').read_text());cfg.pop('expert_preflight',None)
 cfg['evaluation']=dict(cases=inside(out/'evaluation-cases.json'),cases_sha256=sha(out/'evaluation-cases.json'))
 task=cfg['garmi_task'];task['contact_policy']='fingers_and_wrist';task['controller'].update(gripper_open_width_m=.08,alignment_distance=.005,alignment_angle=.05,close_on_approach_timeout=False)
 cfg['expert_precision']=dict(translation_threshold=.002,orientation_threshold=.02)
 task['multi_object'].update(manifest=inside(out/'bank.json'),reference_ids={name:[r['id'] for r in refs]})
 task['kitchen'].update(mask=inside(out/'position-mask.json'),bowl_regions=settings,sampling_mode='validated_subregions')
 task['kitchen']['files_sha256']['mask']=sha(out/'position-mask.json')
 (out/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 save(out/'provenance.json',rows);freeze_repeat(out,conditions,out,out/'task.yaml')
 approach=[]
 for c in conditions:
  approach.append(dict(id=c['id']+'-approach',object=name,reference=c['reference'],object_state=c['object_state'],robot_state=[0.,.05,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]))
 save(out/'approach-cases.json',approach)
 return out


def backup(name):
 master=ROOT/f'resources/validated-ycb-v{4 if name=="mug" else 5}/{name}'
 existing={r['id'] for r in read(master/'bank.json')['objects'][0]['references']}
 refs={r['id']:r for r in read(master/'backup-validated-references.json')}
 rows=[]
 for evidence in read(master/'backup-validation-evidence.json')['references']:
  if evidence['reference'] in existing:continue
  rows.append(dict(ref=refs[evidence['reference']],source=evidence['source'],handle=evidence['handle'],conditions=evidence['conditions']))
 return build(name,'backup',rows)


def new(name):
 source=ROOT/f'runs/ycb-handle-inner-outer-v2/{name}'
 summary=read(OUT/name/'new-initial/physics/summary.json');cases={c['id']:c for c in read(source/'placement-cases.json')}
 refs=read(source/'bank.json')['objects'][0]['references'];rows=[]
 for ref in refs:
  passed=[r for r in summary['results'] if r['reference']==ref['id'] and r['passed']]
  if not passed:continue
  # Retain successful placements; each is revalidated independently at 17 cases.
  conditions=[]
  for r in passed:
   c=cases[r['id']];conditions.append(dict(id=r['id'],reference=r['reference'],region=r['region'],depth=r['depth'],object_state=c['object_state'],base_pose=r['base_pose'],arm_position=r['arm_position'],handle=True))
  rows.append(dict(ref=ref,source=str(source.relative_to(ROOT)),handle=True,conditions=conditions))
 if rows:return build(name,'new-repeat',rows)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['backup','new']);p.add_argument('object',choices=['mug','pitcher']);a=p.parse_args()
 (backup if a.kind=='backup' else new)(a.object)
