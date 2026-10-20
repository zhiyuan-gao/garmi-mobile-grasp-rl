"""Capacity-gated local PPO with BC auxiliary loss, independent object queues."""
import argparse,collections,copy,csv,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rl'))
from garmi_rl.imitation_pipeline import validate_partitions
OUT=ROOT/'runs/wrist-ppo-v6';IMIT=ROOT/'runs/wrist-imitation-v6';DATA=ROOT/'runs/wrist-expert-collection-v6'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inside(p):return '/workspace/'+str(p.relative_to(ROOT))
def host(p):return ROOT/str(p).removeprefix('/workspace/')
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def prepare():
 assert not OUT.exists();OUT.mkdir()
 prior=read(IMIT/'plan.json')
 changed={'rl/garmi_rl/config.py','rl/garmi_rl/mdp.py','rl/garmi_rl/multi_object.py','rl/garmi_rl/trial_evaluation.py','rl/train_trial.py'}
 changes={}
 for rel,digest in prior['code_sha256'].items():
  actual=sha(ROOT/rel)
  if actual!=digest:
   assert rel in changed,rel;changes[rel]=dict(before=digest,after=actual)
 for rel,digest in prior['source_sha256'].items():
  if sha(ROOT/rel)!=digest:assert rel in changes,rel
 plan=dict(status='prepared',ppo_authorized=True,target_updates_per_object=8000,evaluation_interval=500,
  capacity_candidates=[1024,2048,4096,8192],capacity_updates=[2,4,8,16],objects={},source_sha256={},code_sha256={},
  changes_since_imitation=changes,reset_distribution='Uniform reference/trajectory from 12000 audited TRAIN starts only; local task, no full-kitchen generalization',
  checkpoint_selection='Continue round-03 last model; final PPO is update 8000, heldout scores are diagnostics only',
  strict_success='Physical success plus aligned handover AND measured close-start <=5mm/.05rad',
  capacity_policy='Retain successful capacity updates in budget; test 8192 only after 4096 passes and memory reserve permits; failures stop formal training')
 for n in ['pitcher','mug']:
  f=OUT/n;f.mkdir();data=DATA/n;source=IMIT/n;result=read(source/'result.json');assert result['status']=='complete' and result['rounds']==3
  checkpoint=host(result['checkpoint']);assert sha(checkpoint)==result['checkpoint_sha256']
  train,val=read(data/'train-index.json'),read(data/'validation-index.json');heldout=read(source/'evaluation-cases.json')
  refs=validate_partitions(train,val,heldout,n)
  pool=dict(object=n,split='train',distribution='Uniform 300 starts per reference',source_index=inside(data/'train-index.json'),
            source_sha256=sha(data/'train-index.json'),cases=[dict(r['initial_case'],source_initial_id=r['id']) for r in train['demonstrations']])
  write(f/'training-reset-pool.json',pool)
  shutil.copy2(source/'evaluation-cases.json',f/'evaluation-cases.json')
  profile=yaml.safe_load((source/'task.yaml').read_text());profile['garmi_task']['require_strict_success']=True
  profile['garmi_task']['multi_object']['training_reset_pool']=dict(file=inside(f/'training-reset-pool.json'),sha256=sha(f/'training-reset-pool.json'),split='train')
  profile['expert_bc']=dict(coefficient=3.,batch_size=1024,state_normalization='frozen',index=inside(source/'round-03-train-index.json'),data_device='cpu',sampling='uniform_references',loss='action_mse')
  profile['evaluation']=dict(cases=inside(f/'evaluation-cases.json'),cases_sha256=sha(f/'evaluation-cases.json'))
  (f/'ppo.yaml').write_text(yaml.safe_dump(profile,sort_keys=False))
  write(f/'split-audit.json',dict(status='passed',training_reset_cases=12000,references=40,per_reference=300,validation_cases_excluded=1000,heldout_cases_excluded=80))
  plan['objects'][n]=dict(checkpoint=inside(checkpoint),expert_index=inside(source/'round-03-train-index.json'),seed=2026102001 if n=='pitcher' else 2026102002)
  for p in [checkpoint,source/'result.json',source/'round-03-train-index.json',data/'train-index.json',data/'validation-index.json',f/'ppo.yaml',f/'training-reset-pool.json',f/'evaluation-cases.json',f/'split-audit.json']:
   plan['source_sha256'][str(p.relative_to(ROOT))]=sha(p)
  write(f/'supervisor.json',dict(status='prepared',object=n,stages=[],ppo_updates=0))
 for p in list((ROOT/'rl').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/rl.sh',ROOT/'scripts/container_exec.sh']:
  rel=p.relative_to(ROOT);plan['code_sha256'][str(rel)]=sha(p);dest=OUT/'code-snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 write(OUT/'code-snapshot/SHA256.json',plan['code_sha256']);write(OUT/'plan.json',plan);print('PPO_PLAN_PREPARED',flush=True)

def verify(plan):
 for section in ['source_sha256','code_sha256']:
  for rel,digest in plan[section].items():assert sha(ROOT/rel)==digest,rel

def gpu_memory(gpu):
 p=subprocess.run(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return list(map(int,p.stdout.strip().split(',')))

def available_ram():
 return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/1024

def predicted_gpu_memory(capacity,candidate,used):
 """Keep fixed simulator overhead when extrapolating the optional 8192 tier."""
 def measured(row):
  command=row['command']
  return int(command[command.index('--num_envs')+1]),max(0,row['peak_gpu_used_mib']-row['gpu_baseline_mib'])
 current_envs,current_load=measured(capacity[-1])
 if candidate<=4096:return used+current_load*candidate/current_envs
 slopes=[]
 for first,second in zip(capacity,capacity[1:]):
  first_envs,first_load=measured(first);second_envs,second_load=measured(second)
  if second_envs>first_envs:slopes.append(max(0,(second_load-first_load)/(second_envs-first_envs)))
 if not slopes:return used+current_load*candidate/current_envs
 return used+current_load+max(slopes)*(candidate-current_envs)

def run(n,gpu,resume,plan_path=None):
 f=OUT/n;lock=(f/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan_path=plan_path or OUT/'plan.json'
 plan=read(plan_path);verify(plan);state=read(f/'supervisor.json')
 assert state['status']=='prepared' or (resume and state['status']=='failed')
 state.update(status='running',pid=os.getpid(),gpu=gpu,plan_sha256=sha(plan_path),active_plan=str(plan_path.relative_to(ROOT)));state.setdefault('started_unix',time.time());state.pop('error',None);write(f/'supervisor.json',state)
 cfg=f/'ppo.yaml';base=plan['objects'][n]['checkpoint'];seed=plan['objects'][n]['seed']
 def stage(label,script,args,marker):
  verify(plan);command=['bash','scripts/rl.sh','/workspace/rl/'+script]+list(map(str,args))
  old=next((x for x in state['stages'] if x['name']==label),None)
  if old and old['status']=='complete':
   assert old['command']==command
   for rel,digest in old['output_sha256'].items():assert sha(ROOT/rel)==digest,rel
   return old
  if old:
   assert resume
   archive=f/f'recovery-{int(time.time())}-{label}';archive.mkdir()
   for p in [f/label,f/(label+'.log')]:
    if p.exists():shutil.move(str(p),str(archive/p.name))
   state['stages'].remove(old)
  row=dict(name=label,status='running',command=command,started_unix=time.time(),gpu_baseline_mib=gpu_memory(gpu)[0],peak_gpu_used_mib=0,minimum_available_ram_mib=available_ram())
  state['stages'].append(row);state['current_stage']=label;write(f/'supervisor.json',state)
  with (f/(label+'.log')).open('w') as log:
   child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,GARMI_RL_GPU=gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   row['pid']=child.pid;write(f/'supervisor.json',state)
   try:
    while child.poll() is None:
     used,total=gpu_memory(gpu);row['peak_gpu_used_mib']=max(row['peak_gpu_used_mib'],used);row['gpu_total_mib']=total;row['minimum_available_ram_mib']=min(row['minimum_available_ram_mib'],available_ram())
     write(f/'supervisor.json',state)
     if time.time()-row['started_unix']>172800:raise TimeoutError(label)
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:pass
   except BaseException:
    if child.poll() is None:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=45)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
  text=(f/(label+'.log')).read_text(errors='replace');bad=[x for x in ['Traceback (most recent call last)','CUDA error:','out of memory','buffer overflow','PhysX error'] if x.lower() in text.lower()]
  row.update(seconds=time.time()-row['started_unix'],returncode=child.returncode,errors=bad,status='complete' if child.returncode==0 and marker in text and not bad else 'failed')
  dest=host(command[command.index('--out')+1]);paths=[dest] if dest.is_file() else [dest/x for x in ['summary.json','latest.json'] if (dest/x).exists()]
  if (dest/'latest.json').exists():
   cp=host(read(dest/'latest.json')['checkpoint']);paths += [cp,cp.with_suffix('.resume.pt')]
  row['output_sha256']={str(p.relative_to(ROOT)):sha(p) for p in paths};write(f/'supervisor.json',state)
  assert row['status']=='complete',f'{label} failed: {bad}; see {f/(label+".log")}'
  print('PPO_STAGE_DONE',n,label,round(row['seconds'],1),flush=True);return row
 def train(label,envs,updates,checkpoint=None,previous_envs=None):
  args=['--headless','--config',inside(cfg),'--out',inside(f/label),'--num_envs',envs,'--updates',updates,
        '--eval-updates',updates,'--skip-evaluation','--seed',seed]
  args+=['--resume',checkpoint] if checkpoint else ['--init-checkpoint',base]
  if checkpoint and previous_envs!=envs:args+=['--allow-env-resize']
  row=stage(label,'train_trial.py',args,'TRIAL_DONE');summary=read(f/label/'summary.json')
  assert summary['status']=='complete' and summary['update']==updates and summary['frozen_state_normalization_verified']
  state.update(ppo_updates=updates,num_envs=envs,checkpoint=summary['latest_checkpoint']);write(f/'supervisor.json',state)
  return summary['latest_checkpoint'],row
 def audit(label,checkpoint,target):
  stage(label+'-audit','audit_ppo_transfer.py',['--bc',base,'--checkpoint',checkpoint,'--stage',inside(f/label),
        '--expected-update',target,'--out',inside(f/(label+'-audit.json'))],'PPO_TRANSFER_AUDIT_PASSED')
  assert read(f/(label+'-audit.json'))['status']=='passed'
 def evaluate(label,checkpoint,target):
  stage(label,'evaluate_kitchen.py',['--headless','--config',inside(cfg),'--out',inside(f/label),
       '--num_envs',80,'--checkpoint',checkpoint,'--require-strict-success','--seed',12345],'EVALUATION_DONE')
  s=read(f/label/'summary.json');assert s['status']=='complete' and s['completed']==80 and s['require_strict_success']
  assert s['evaluation_cases_sha256']==sha(f/'evaluation-cases.json') and s['checkpoint_sha256']==sha(host(checkpoint))
  assert all(x['strict_success_verified'] for x in s['results'] if x['reason']=='success')
  row=dict(update=target,checkpoint=checkpoint,success_rate=s['success_rate'],counts=s['counts'],cases=80)
  existing=read(f/'evaluations.json') if (f/'evaluations.json').exists() else []
  existing=[x for x in existing if x['update']!=target]+[row];write(f/'evaluations.json',existing)
 try:
  stage('reset-check','check_local_ppo.py',['--headless','--config',inside(cfg),'--out',inside(f/'reset-check'),'--num_envs',256],'LOCAL_PPO_RESET_PASSED')
  assert read(f/'reset-check/summary.json')['status']=='passed'
  checkpoint=None;envs=None
  # Decide only on resource measurements, never on held-out success rates.
  capacity=[];decisions=[]
  for candidate,target in zip(plan['capacity_candidates'],plan['capacity_updates']):
   if candidate>4096 and envs!=4096:
    decisions.append(dict(candidate=candidate,status='skipped',reason='4096 environments not validated'))
    break
   selected=candidate
   predicted=None
   if capacity:
    used,total=gpu_memory(gpu)
    predicted=predicted_gpu_memory(capacity,candidate,used)
    if predicted>total-4096 or available_ram()<32768:selected=envs
   label=f'capacity-{target:04d}'
   # Recovery must reuse the originally recorded resource choice.
   old=next((s for s in state['stages'] if s['name']==label),None)
   if old:selected=int(old['command'][old['command'].index('--num_envs')+1])
   if candidate>4096 and selected!=candidate and not old:
    decisions.append(dict(candidate=candidate,status='skipped',reason='memory reserve',
                          predicted_gpu_mib=predicted,available_ram_mib=available_ram()))
    break
   checkpoint,row=train(label,selected,target,checkpoint,envs);envs=selected;capacity.append(row)
   audit(label,checkpoint,target)
   decisions.append(dict(candidate=candidate,num_envs=selected,status='passed',update=target,predicted_gpu_mib=predicted))
  capacity_update=state['ppo_updates']
  evaluate('capacity-evaluation',checkpoint,capacity_update)
  write(f/'capacity-report.json',dict(status='passed',num_envs=envs,updates=capacity_update,decisions=decisions,optimizer_resume_verified=True,
        frozen_normalization_verified=True,strict_evaluation_verified=True,capacity_stages=[{k:x[k] for k in ['name','peak_gpu_used_mib','minimum_available_ram_mib','seconds']} for x in capacity]))
  for target in range(500,8001,500):
   label=f'ppo-{target:04d}';checkpoint,row=train(label,envs,target,checkpoint,envs)
   audit(label,checkpoint,target);evaluate(f'evaluation-{target:04d}',checkpoint,target)
  state.update(status='complete',checkpoint=checkpoint)
  write(f/'result.json',dict(status='complete',updates=8000,num_envs=envs,checkpoint=checkpoint,
        checkpoint_sha256=sha(host(checkpoint)),evaluations=read(f/'evaluations.json')))
 except BaseException as e:state.update(status='failed',error=repr(e));raise
 finally:
  state['updated_unix']=time.time();write(f/'supervisor.json',state);print('PPO_SUPERVISOR',n,state['status'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['prepare','run']);p.add_argument('--object',choices=['pitcher','mug']);p.add_argument('--gpu',choices=['0','1']);p.add_argument('--resume',action='store_true');p.add_argument('--plan',type=Path);a=p.parse_args()
 if a.mode=='prepare':prepare()
 else:
  assert a.object and a.gpu is not None;run(a.object,a.gpu,a.resume,a.plan.resolve() if a.plan else None)
