"""Independent pitcher/mug BC and three corrective-imitation rounds; no PPO entrypoint."""
import argparse
import collections
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'rl'))
from garmi_rl.imitation_pipeline import (validate_partitions, sample_correction_cases,
                                        teacher_arguments, aggregate_labels)

OUT = ROOT/'runs/wrist-imitation-v6'
DATA = ROOT/'runs/wrist-expert-collection-v6'
CRAM = '/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def inside(path): return '/workspace/'+str(path.relative_to(ROOT))
def host(path): return ROOT/str(path).removeprefix('/workspace/')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def prepare():
    assert not OUT.exists(), 'Never overwrite an existing training run'
    OUT.mkdir()
    plan = dict(status='prepared', bc_authorized=True, corrective_imitation_authorized=True,
        ppo_authorized=False, objects={}, rounds=3, bc_epochs=100, batch_size=1024,
        learning_rate=.0003, reference_sampling='uniform_references', workers=12,
        correction_envs=200, correction_train_cases=2000, correction_validation_cases=400,
        correction_probability=.5, correction_steps=[5,600], checkpoint_policy='last epoch, as repository pipeline',
        initialization='Independent fresh networks; smoke models never initialize formal training',
        distribution='Collected local starts; no claim of full-kitchen generalization',
        evaluation_policy='80 frozen held-out cases per object, evaluation only; never used for labels or model selection',
        code_sha256={}, source_sha256={})
    # Verify the completed collection's frozen physics, data sources and runtime before adapting training.
    collection_plan = read(DATA/'parallel-resize-v1/plan.json')
    for section in ['code_sha256','source_sha256']:
        for rel,digest in collection_plan[section].items():
            assert sha(ROOT/rel) == digest, rel
            plan['source_sha256'][rel] = digest
    for name, seed in [('pitcher',2026101983),('mug',2026101984)]:
        source=DATA/name; folder=OUT/name; folder.mkdir()
        assert read(source/'supervisor.json')['status']=='complete'
        assert read(source/'data-audit.json')['status']=='passed'
        train, validation = [read(source/f'{split}-index.json') for split in ['train','validation']]
        heldout = read(source/'evaluation-cases.json')
        refs = validate_partitions(train, validation, heldout, name)
        assert len(heldout)==80
        profile=yaml.safe_load((source/'task.yaml').read_text())
        assert profile['garmi_task']['multi_object']['objects']==[name]
        assert profile['garmi_task']['contact_policy']=='fingers_and_wrist'
        assert profile['garmi_task']['controller']['close_on_approach_timeout'] is False
        profile['agent']['experiment']['experiment_name']=name+'_wrist_imitation_v6'
        shutil.copy2(source/'evaluation-cases.json',folder/'evaluation-cases.json')
        profile['evaluation']=dict(cases=inside(folder/'evaluation-cases.json'),cases_sha256=sha(folder/'evaluation-cases.json'))
        (folder/'task.yaml').write_text(yaml.safe_dump(profile,sort_keys=False))
        for split,data in [('train',train),('validation',validation)]:
            pools=collections.defaultdict(list)
            for row in data['demonstrations']:pools[row['reference']].append(row)
            subset=[row for ref in refs for row in pools[ref][:2 if split=='train' else 1]]
            write(folder/f'smoke-{split}-index.json',dict(data,demonstrations=subset,trajectories=len(subset),
                  scope='Interface smoke subset; not the formal training dataset'))
            cases=sorted(sample_correction_cases(dict(demonstrations=subset),1,seed),key=lambda c:c['reference'])[:8]
            write(folder/f'smoke-{split}-cases.json',cases)
        write(folder/'smoke-evaluation-cases.json',heldout[:8])
        validation_cases=sample_correction_cases(validation,10,seed+1)
        for rnd in range(1,4):
            training_cases=sample_correction_cases(train,50,seed+100*rnd)
            for split,cases in [('train',training_cases),('validation',validation_cases)]:
                for offset in range(0,len(cases),200):
                    write(folder/f'round-{rnd:02d}-{split}-{offset//200:02d}-cases.json',cases[offset:offset+200])
        write(folder/'split-audit.json',dict(status='passed',references=refs,train=12000,validation=1000,
            heldout=80,disjoint_initial_states=True,correction_parents_preserve_original_split=True,
            correction_train_per_round=2000,correction_validation_per_round=400))
        plan['objects'][name]=dict(seed=seed,teacher_arguments=teacher_arguments(profile),
            train_index=inside(source/'train-index.json'),validation_index=inside(source/'validation-index.json'))
        for path in list(folder.glob('*.json'))+[folder/'task.yaml',source/'train-index.json',
                source/'validation-index.json',source/'data-audit.json',source/'supervisor.json']:
            plan['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
        write(folder/'supervisor.json',dict(status='prepared',object=name,stages=[],ppo_started=False))
    paths=list((ROOT/'rl').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/rl.sh',ROOT/'scripts/container_exec.sh']
    for path in paths:
        if '__pycache__' in path.parts:continue
        rel=path.relative_to(ROOT);plan['code_sha256'][str(rel)]=sha(path)
        target=OUT/'code-snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write(OUT/'code-snapshot/SHA256.json',plan['code_sha256'])
    write(OUT/'plan.json',plan)
    print('IMITATION_PLAN_PREPARED',flush=True)


def verify(plan):
    assert plan['ppo_authorized'] is False
    for section in ['source_sha256','code_sha256']:
        for rel,digest in plan[section].items():assert sha(ROOT/rel)==digest,rel


def run(name,gpu,phase,resume):
    folder=OUT/name
    lock=(folder/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(OUT/'plan.json');verify(plan)
    state=read(folder/'supervisor.json')
    if phase=='formal':assert state['status']=='smoke_passed' or (resume and state['status']=='failed')
    else:assert state['status']=='prepared' or (resume and state['status']=='failed')
    state.update(status='running',pid=os.getpid(),gpu=gpu,phase=phase,updated_unix=time.time())
    state.setdefault('started_unix',time.time())
    state.pop('error',None)
    state['plan_sha256']=sha(OUT/'plan.json')
    write(folder/'supervisor.json',state)
    cfg=folder/'task.yaml';obj=plan['objects'][name];seed=obj['seed']

    def stage(label,script,args,marker,cpu=False):
        verify(plan)
        command=['bash','scripts/container_exec.sh',CRAM] if cpu else ['bash','scripts/rl.sh']
        command += ['/workspace/rl/'+script]+list(map(str,args))
        existing=next((s for s in state['stages'] if s['name']==label),None)
        input_hashes={}
        for flag in ['--config','--index','--validation','--init-checkpoint','--checkpoint','--cases','--queries','--expert-results','--requests']:
            if flag in command:input_hashes[flag]=sha(host(command[command.index(flag)+1]))
        if existing and existing['status']=='complete':
            assert existing['command']==command and existing['input_sha256']==input_hashes
            for rel,digest in existing['output_sha256'].items():assert sha(ROOT/rel)==digest,rel
            return
        if existing:
            assert resume
            archive=folder/f'recovery-{int(time.time())}-{label}';archive.mkdir()
            for path in [folder/label,folder/(label+'.log')]:
                if path.exists():shutil.move(str(path),str(archive/path.name))
            state['stages'].remove(existing)
        row=dict(name=label,status='running',command=command,input_sha256=input_hashes,started_unix=time.time())
        state['stages'].append(row);state.update(current_stage=label,updated_unix=time.time())
        write(folder/'supervisor.json',state)
        log_path=folder/(label+'.log')
        with log_path.open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,GARMI_RL_GPU=gpu),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row['pid']=child.pid;write(folder/'supervisor.json',state)
            try:rc=child.wait(timeout=86400)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        text=log_path.read_text(errors='replace')
        bad=[t for t in ['Traceback (most recent call last)','CUDA error:','out of memory','buffer overflow','PhysX error'] if t.lower() in text.lower()]
        row.update(returncode=rc,seconds=time.time()-row['started_unix'],errors=bad,
                   status='complete' if rc==0 and marker in text and not bad else 'failed')
        output=host(command[command.index('--out')+1])
        artifacts=([output] if output.is_file() else [output/f for f in
            ['summary.json','index.json','last.pt','best.pt','labels.npz','dagger-queries.json'] if (output/f).exists()])
        row['output_sha256']={str(p.relative_to(ROOT)):sha(p) for p in artifacts}
        state['updated_unix']=time.time();write(folder/'supervisor.json',state)
        assert row['status']=='complete', f'{label} failed; inspect {log_path}'
        print('IMITATION_STAGE_DONE',name,label,round(row['seconds'],1),flush=True)

    def bc(label,train,val,checkpoint=None,epochs=100):
        args=['--config',inside(cfg),'--index',inside(train),'--validation',inside(val),'--out',inside(folder/label),
              '--epochs',epochs,'--patience',1000,'--batch-size',1024,'--learning-rate',.0003,
              '--loss','action_mse','--balance-references','--seed',seed]
        if checkpoint:args+=['--init-checkpoint',checkpoint]
        stage(label,'train_expert_bc.py',args,'BC_DONE')
        summary=read(folder/label/'summary.json')
        assert summary['status']=='complete' and summary['epoch']==epochs
        assert summary['final']['checkpoint_reload_max_error']==0
        assert sha(folder/label/'last.pt')==summary['final']['sha256']
        return inside(folder/label/'last.pt')

    def evaluate(label,checkpoint,cases):
        stage(label,'evaluate_kitchen.py',['--headless','--config',inside(cfg),'--checkpoint',checkpoint,
            '--cases',inside(cases),'--num_envs',80,'--seed',seed,'--require-strict-success','--out',inside(folder/label)],'EVALUATION_DONE')
        s=read(folder/label/'summary.json')
        assert s['status']=='complete' and s['completed']==len(read(cases)) and s['require_strict_success']
        assert s['checkpoint_sha256']==sha(host(checkpoint)) and s['evaluation_cases_sha256']==sha(cases)
        assert all(r['object']==name and (r['reason']!='success' or r['strict_success_verified']) for r in s['results'])
        return dict(stage=label,success_rate=s['success_rate'],counts=s['counts'],cases=s['completed'])

    def correction(label,checkpoint,cases,smoke=False):
        args=['--headless','--config',inside(cfg),'--checkpoint',checkpoint,'--cases',inside(cases),
            '--num_envs',200,'--out',inside(folder/label),'--seed',seed+len(state['stages']),
            '--dagger-queries','--dagger-stochastic','--giskard-stages','--require-strict-success',
            '--dagger-probability',.5,'--dagger-min-step',5,'--dagger-max-step',18 if smoke else 600]
        if smoke:args+=['--dagger-stop-after-queries']
        # Fixed seed tied to stage label rather than stage count makes interrupted-stage resume reproducible.
        args[args.index('--seed')+1]=seed+int(hashlib.sha256(label.encode()).hexdigest()[:6],16)
        stage(label,'evaluate_kitchen.py',args,'EVALUATION_DONE')
        summary=read(folder/label/'summary.json')
        assert summary['status']=='complete' and summary['dagger']['queries']>0
        assert summary['dagger']['expert_actions_executed']==0
        queries=folder/label/'dagger-queries.json';answers=folder/(label+'-expert.json')
        stage(label+'-expert','giskard_expert_service.py',['--requests',inside(queries),'--out',inside(answers),
            '--workers',12]+obj['teacher_arguments'],'GISKARD_BATCH_DONE',cpu=True)
        stage(label+'-labels','build_giskard_labels.py',['--queries',inside(queries),
            '--expert-results',inside(answers),'--out',inside(folder/(label+'-labels'))],'GISKARD_LABELS_DONE')
        labels=read(folder/(label+'-labels')/'index.json')
        assert labels['status']=='passed' and labels['labels']>0 and labels['maximum_frame_error']<3.e-4
        assert labels['expert_actions_executed'] is False
        return labels['dagger_label_sets']

    try:
        if phase=='smoke':
            train,val=[folder/f'smoke-{s}-index.json' for s in ['train','validation']]
            checkpoint=bc('smoke-bc',train,val,epochs=2)
            for split,base in [('train',train),('validation',val)]:
                labels=correction('smoke-'+split,checkpoint,folder/f'smoke-{split}-cases.json',smoke=True)
                write(folder/f'smoke-aggregated-{split}.json',aggregate_labels(read(base),labels))
            corrected=bc('smoke-corrected-bc',folder/'smoke-aggregated-train.json',
                         folder/'smoke-aggregated-validation.json',checkpoint,epochs=1)
            report=evaluate('smoke-evaluation',corrected,folder/'smoke-evaluation-cases.json')
            write(folder/'smoke-report.json',dict(status='passed',interface_only=True,evaluation=report,
                note='Tiny subset/epochs validate plumbing only; success rate is not a trained-policy result',
                no_expert_actions_executed=True,ppo_started=False))
            state['status']='smoke_passed'
        else:
            assert read(folder/'smoke-report.json')['status']=='passed'
            train,val=host(obj['train_index']),host(obj['validation_index'])
            checkpoint=bc('initial-bc',train,val)
            evaluations=[evaluate('initial-bc-evaluation',checkpoint,folder/'evaluation-cases.json')]
            accumulated={'train':[],'validation':[]}
            for rnd in range(1,4):
                prefix=f'round-{rnd:02d}'
                for split,chunks,base in [('train',10,train),('validation',2,val)]:
                    for chunk in range(chunks):
                        label=f'{prefix}-{split}-{chunk:02d}'
                        accumulated[split]+=correction(label,checkpoint,folder/(label+'-cases.json'))
                    write(folder/(prefix+f'-{split}-index.json'),aggregate_labels(read(base),accumulated[split]))
                checkpoint=bc(prefix+'-bc',folder/(prefix+'-train-index.json'),
                              folder/(prefix+'-validation-index.json'),checkpoint)
                evaluations.append(evaluate(prefix+'-evaluation',checkpoint,folder/'evaluation-cases.json'))
                write(folder/'evaluations.json',evaluations)
            write(folder/'result.json',dict(status='complete',checkpoint=checkpoint,checkpoint_sha256=sha(host(checkpoint)),
                evaluations=evaluations,rounds=3,dagger_labels={s:sum(x['count'] for x in v) for s,v in accumulated.items()},
                ppo_started=False))
            state.update(status='complete',checkpoint=checkpoint)
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:
        state['updated_unix']=time.time();write(folder/'supervisor.json',state)
        print('IMITATION_SUPERVISOR',name,state['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--object',choices=['pitcher','mug']);p.add_argument('--gpu',choices=['0','1'])
    p.add_argument('--phase',choices=['smoke','formal'],default='smoke');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:
        assert a.object and a.gpu is not None
        run(a.object,a.gpu,a.phase,a.resume)
