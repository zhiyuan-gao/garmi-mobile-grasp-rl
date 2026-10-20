"""Freeze new wrist-contact policy validation inputs. No inherited pass labels."""
import argparse,copy,json,random
from pathlib import Path
import numpy as np,yaml
from select_references import ROOT,sha
from prepare_rl_validation import save
from prepare_grasp_repeatability import CRITERIA
from try_handle_roll import read,host,inside
OUT=ROOT/'runs/wrist-contact-v6'

def configure(source,out):
 out.mkdir(parents=True,exist_ok=True)
 cfg=yaml.safe_load((source/'task.yaml').read_text());cfg.pop('expert_preflight',None)
 task=cfg['garmi_task'];task['contact_policy']='fingers_and_wrist'
 task['controller'].update(alignment_distance=.005,alignment_angle=.05,close_on_approach_timeout=False)
 cfg['expert_precision']=dict(translation_threshold=.002,orientation_threshold=.02)
 (out/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 return cfg

def freeze_repeat(out,conditions,source,config_path):
 out.mkdir(parents=True,exist_ok=True);save(out/'criteria.json',CRITERIA)
 cases=[];mask=read(source/'position-mask.json')
 variants=[('nominal',[0.,0.])]*5+[('object_offset',p) for p in CRITERIA['object_offsets_m']]+[('base_offset',p) for p in CRITERIA['base_offsets_m']]
 name=mask['object_name']
 for c in conditions:
  for i,(kind,offset) in enumerate(variants):
   case=dict(id=c['id']+f'-v6-{i:02d}',condition_id=c['id'],object=name,reference=c['reference'],region=c['region'],depth=c['depth'],
    object_state=copy.deepcopy(c['object_state']),robot_state=[0.,.05,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],endpoint_base_pose=copy.deepcopy(c['base_pose']),endpoint_arm_seed=copy.deepcopy(c['arm_position']),test_kind=kind,offset_task_xy_m=offset)
   if kind=='object_offset':case['object_state'][:2]=(np.array(case['object_state'][:2])+offset).tolist()
   if kind=='base_offset':case['endpoint_base_pose'][:2]=(np.array(case['endpoint_base_pose'][:2])+offset).tolist()
   xy=np.array([.3-case['object_state'][1],6.35+case['object_state'][0]]);idx=mask['reference_ids'].index(c['reference'])
   assert any((xy>=np.array(cell['bounds_map_xy'][0])-1e-7).all() and (xy<=np.array(cell['bounds_map_xy'][1])+1e-7).all() and idx in cell['allowed_reference_indices'] for cell in mask['cells'])
   cases.append(case)
 random.Random(2026101901).shuffle(cases)
 save(out/'cases.json',cases);save(out/'conditions.json',conditions)
 save(out/'inputs.json',dict(source=str(source.relative_to(ROOT)),configuration=str(config_path.relative_to(ROOT)),configuration_sha256=sha(config_path),
  cases_sha256=sha(out/'cases.json'),criteria_sha256=sha(out/'criteria.json'),source_bank_sha256=sha(source/'bank.json'),case_count=len(cases),reference_count=len({c['reference'] for c in conditions}),
  contact_policy='fingers_and_wrist',old_success_labels_inherited=False))
 print('FROZEN',out.name,len(conditions),'conditions',len(cases),'cases',flush=True)

def initial():
 OUT.mkdir(exist_ok=True)
 for name,version in [('mug',4),('pitcher',5)]:
  old=ROOT/f'resources/validated-ycb-v{version}/{name}'
  target=OUT/name/'existing';configure(old,target)
  conditions=read(old/'endpoint-conditions.json')['conditions']
  freeze_repeat(target,conditions,old,target/'task.yaml')
  new=ROOT/f'runs/ycb-handle-inner-outer-v2/{name}'
  configure(new,OUT/name/'new-initial')
 save(OUT/'criteria.json',dict(local_repeatability=CRITERIA,contact_policy='fingers_and_wrist',
  target_count_per_object=40,expert_handover_position_m=.005,expert_handover_angle_rad=.05,expert_translation_stop_m=.002,expert_rotation_stop_rad=.02,
  no_forced_close_on_timeout=True,other_contact_and_success_thresholds_unchanged=True))

if __name__=='__main__':initial()
