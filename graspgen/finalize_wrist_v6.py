"""Package 40 distinct references only after both new-policy validation gates pass."""
import argparse
import copy
import shutil
import numpy as np
import yaml
from prepare_rl_validation import save
from prepare_wrist_v6 import OUT
from review_wrist_v6 import main as review, matching_condition, strict_approach
from select_references import ROOT, sha, pose_components, distances_to, coverage
from try_handle_roll import read
from wrist_v6_inputs import build


def configured_object(folder):
    cfg=yaml.safe_load((folder/'task.yaml').read_text())
    bank_path=ROOT/cfg['garmi_task']['multi_object']['manifest'].removeprefix('/workspace/')
    return cfg,read(bank_path)['objects'][0]


def retain_all_placement_proofs(name):
    """Each published placement window retains a matching full-approach trace."""
    folder=ROOT/'resources/validated-ycb-v6'/name
    report=read(folder/'validation.json');status=read(OUT/'status.json')['objects'][name]
    for c in report['references']:
        proofs=[c['approach_evidence']];extra=[]
        for condition in c['conditions']:
            if any(matching_condition(condition,p['result']) for p in proofs):continue
            proof=copy.deepcopy(next(p for p in status['approach_evidence'][c['ref']['id']] if matching_condition(condition,p['result'])))
            row=proof['result'];assert strict_approach(row)
            source=ROOT/row['file'].removeprefix('/workspace/');assert sha(source)==row['sha256']
            destination=folder/'validation-traces'/f"{c['ref']['id']}-placement-{len(extra)+2}.npz"
            shutil.copy2(source,destination);proof['saved_trace']=str(destination.relative_to(ROOT))
            proof['summary_sha256']=sha(ROOT/proof['summary'])
            proof['runtime_code']=read((ROOT/proof['summary']).parent/'runtime-code.json')
            extra.append(proof);proofs.append(proof)
        c['additional_approach_evidence']=extra
    save(folder/'validation.json',report)


def finalize(name):
    review()
    runtime_code=read(OUT/'code-planner-v5/SHA256.json')
    for path,digest in runtime_code.items():assert sha(ROOT/path)==digest, path
    status=read(OUT/'status.json')['objects'][name]
    assert status['both_passed']>=40, (name,status['both_passed'])
    candidates=[]
    for ref in status['accepted_reference_ids']:
        endpoint=status['endpoint_evidence'][ref]
        approaches=status['approach_evidence'][ref]
        endpoint=[e for e in endpoint if any(matching_condition(e['condition'],a['result']) for a in approaches)]
        source=endpoint[0]['source']
        endpoint=[e for e in endpoint if e['source']==source]
        obj=read(ROOT/source/'bank.json')['objects'][0]
        reference=copy.deepcopy(next(r for r in obj['references'] if r['id']==ref))
        success=next(a for a in approaches if matching_condition(endpoint[0]['condition'],a['result']))
        assert strict_approach(success['result'])
        ec,eo=configured_object(ROOT/endpoint[0]['folder'])
        ac,ao=configured_object((ROOT/success['summary']).parent.parent)
        assert {k:v for k,v in eo.items() if k!='references'}=={k:v for k,v in ao.items() if k!='references'}
        er=next(r for r in eo['references'] if r['id']==ref)
        ar=next(r for r in ao['references'] if r['id']==ref)
        for key in ['position','quaternion','mesh_T_gripper']:
            assert np.allclose(er[key],ar[key],rtol=0,atol=1e-8)
            assert np.allclose(er[key],reference[key],rtol=0,atol=1e-8)
        ew=er.get('opening_m',ec['garmi_task']['controller'].get('gripper_open_width_m',.08))
        aw=ar.get('opening_m',ac['garmi_task']['controller'].get('gripper_open_width_m',.08))
        assert abs(ew-aw)<1e-8
        reference['opening_m']=ew
        candidates.append(dict(ref=reference,source=source,handle=endpoint[0]['condition']['handle'],
            conditions=[e['condition'] for e in endpoint],endpoint_evidence=endpoint,approach_evidence=success))
    poses=np.asarray([c['ref']['mesh_T_gripper'] for c in candidates]);centers,rotations=pose_components(poses)
    handles=np.array([c['handle'] for c in candidates],bool)
    novel=np.array(['innerouter' in c['ref']['id'] for c in candidates],bool)
    scores=np.array([c['ref'].get('score',0.) for c in candidates])
    nearest=np.full(len(candidates),np.inf);selected=[]

    def choose(allowed):
        allowed=allowed.copy();allowed[selected]=False
        allowed &= nearest>1e-4  # Includes the two-finger swap symmetry.
        assert allowed.any(), 'Not enough distinct successful references'
        i=int(np.argmax(np.where(allowed,nearest if selected else scores,-np.inf)))
        selected.append(i);nearest[:]=np.minimum(nearest,distances_to(centers,rotations,i))

    assert novel.any(), 'Requested inner/outer handle grasp has not passed both gates'
    choose(novel)
    if name=='mug':
        handle_target=min(10,int(handles.sum()))
        assert handle_target>=8, 'Too few validated handle grasps for the requested coverage'
        while sum(handles[selected])<handle_target:choose(handles)
        while len(selected)<40:choose(~handles)
    else:
        while len(selected)<40:choose(np.ones(len(candidates),bool))
    rows=[candidates[i] for i in selected]
    source=OUT/name/'final-inputs'
    assert not source.exists(), 'Do not overwrite a published/frozen selection'
    build(name,'final-inputs',rows)
    target=ROOT/'resources/validated-ycb-v6'/name
    target.mkdir(parents=True,exist_ok=False)
    for file in ['bank.json','conditions.json','position-mask.json','task.yaml','approach-cases.json']:
        shutil.copy2(source/file,target/file)
    bank=read(target/'bank.json');bank['status']='40_distinct_references_passed_local_and_strict_approach_validation'
    save(target/'bank.json',bank);save(target/'references.json',bank['objects'][0]['references'])
    save(target/'endpoint-conditions.json',dict(conditions=read(target/'conditions.json')))
    mask=read(target/'position-mask.json')
    mask['bank_path']=str((target/'bank.json').relative_to(ROOT))
    mask['sources_sha256'].pop(str((source/'bank.json').relative_to(ROOT)))
    mask['sources_sha256'][str((target/'bank.json').relative_to(ROOT))]=sha(target/'bank.json')
    save(target/'position-mask.json',mask)
    # Audit cases are previously successful diagnostic starts, never a held-out evaluation set.
    audits=[copy.deepcopy(c['approach_evidence']['result']['initial_case']) for c in rows]
    save(target/'audit-cases.json',audits);save(target/'evaluation-cases.json',[])
    cfg=yaml.safe_load((source/'task.yaml').read_text())
    old='/workspace/'+str(source.relative_to(ROOT));new='/workspace/'+str(target.relative_to(ROOT))
    def rebind(value):
        if isinstance(value,dict):return {k:rebind(v) for k,v in value.items()}
        if isinstance(value,list):return [rebind(v) for v in value]
        if isinstance(value,str) and value.startswith(old+'/'):return new+value[len(old):]
        return value
    cfg=rebind(cfg)
    cfg['expert_collision']=dict(model='usd_adjacent',self_buffer_m=.005,distal_buffer_m=.003 if name=='mug' else .05)
    cfg['agent']['experiment']['experiment_name']=f'{name}_wrist_v6'
    cfg['garmi_task']['kitchen']['files_sha256']['mask']=sha(target/'position-mask.json')
    cfg['evaluation']['cases_sha256']=sha(target/'evaluation-cases.json')
    (target/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    trajectories=target/'validation-traces';trajectories.mkdir()
    for c in rows:
        e=c['approach_evidence'];r=e['result']
        src=ROOT/r['file'].removeprefix('/workspace/');dest=trajectories/(r['reference']+'.npz')
        assert sha(src)==r['sha256'];shutil.copy2(src,dest)
        e['saved_trace']=str(dest.relative_to(ROOT))
        e['summary_sha256']=sha(ROOT/e['summary'])
        e['runtime_code']=read((ROOT/e['summary']).parent/'runtime-code.json')
        assert sha(ROOT/e['runtime_code']['snapshot'])==e['runtime_code']['snapshot_sha256']
        for local in c['endpoint_evidence']:
            local['review_sha256']=sha(ROOT/local['folder']/'review.json')
        assert sha(dest)==r['sha256']
    report=dict(status='40_passed_packaged_pending_runtime_path_check',object=name,reference_count=40,
        handle_count=sum(c['handle'] for c in rows),inner_outer_handle_count=sum('innerouter' in c['ref']['id'] for c in rows),
        contact_policy='fingers_and_wrist',criteria=read(OUT/'criteria.json'),
        coverage=coverage(poses,np.array(selected)),selection='Farthest pose coverage, including a validated inner/outer handle grasp; mug about 10 handle references',
        scope='Local +/-5 mm placement repeatability and at least one successful full approach from a legal parked-arm start per reference; not a random-reset success-rate guarantee',
        configuration_sha256=sha(target/'task.yaml'),references=rows,training_started=False,
        validation_traces_are_training_dataset=False,runtime_code_sha256=runtime_code)
    save(target/'validation.json',report)
    # A lightweight final runtime test checks the published indexing and mixed finger openings.
    cases=[]
    for i,c in enumerate(rows):
        condition=c['conditions'][0]
        original=next(x for x in read(source/'cases.json') if x['condition_id']==condition['id'] and x['test_kind']=='nominal')
        case=copy.deepcopy(original);case['id']=f'{name}-published-v6-{i:02d}';cases.append(case)
    save(target/'runtime-cases.json',cases)
    retain_all_placement_proofs(name)
    print('PACKAGED_PENDING_RUNTIME_CHECK',target,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('object',choices=['mug','pitcher']);a=p.parse_args();finalize(a.object)
