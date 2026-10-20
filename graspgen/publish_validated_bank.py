"""Copy a reviewed 40-reference bank into stable project resources.

Rebind paths and hashes; never invoke an expert planner or training. The copied
configuration requires its own reset/restore check after publication.
"""
import argparse
import json
import shutil
from pathlib import Path
import yaml
from select_references import ROOT,sha
from prepare_rl_validation import save


def main(name,root=None,target=None):
    root=(root or ROOT/'runs/validated-reference-bank-v4').resolve()
    source=root/'final'/name/'expert-inputs'
    target=(target or ROOT/'resources/validated-ycb-v4'/name).resolve()
    review=source.parent/'review.json'
    check=json.loads((source/'reset-check/summary.json').read_text())
    assert check['status']=='passed' and check['configuration_sha256']==sha(source/'task.yaml')
    bank=json.loads((source/'bank.json').read_text())
    refs=bank['objects'][0]['references'];assert len(refs)==40
    target.mkdir(parents=True,exist_ok=False)
    for file in ['bank.json','endpoint-conditions.json','audit-cases.json','evaluation-cases.json',
                 'retained-grasps.png','preparation.json']:
        shutil.copy2(source/file,target/file)
    for file in ['coverage.png','reference-validation.csv']:
        if (source/file).exists():shutil.copy2(source/file,target/file)
    shutil.copy2(review,target/'review.json')
    shutil.copy2(root/name/'all-validated-references.json',target/'backup-validated-references.json')
    shutil.copy2(root/name/'validation-evidence.json',target/'backup-validation-evidence.json')
    save(target/'references.json',refs)
    mask=json.loads((source/'position-mask.json').read_text())
    mapping={str((source/file).relative_to(ROOT)):str((target/file).relative_to(ROOT))
             for file in ['bank.json','endpoint-conditions.json']}
    mapping[str(review.relative_to(ROOT))]=str((target/'review.json').relative_to(ROOT))
    mask['bank_path']=str((target/'bank.json').relative_to(ROOT))
    mask['sources_sha256']={mapping.get(path,path):digest for path,digest in mask['sources_sha256'].items()}
    for path,digest in mask['sources_sha256'].items():assert sha(ROOT/path)==digest
    save(target/'position-mask.json',mask)
    cfg=yaml.safe_load((source/'task.yaml').read_text())
    old='/workspace/'+str(source.relative_to(ROOT));new='/workspace/'+str(target.relative_to(ROOT))
    def rebind(value):
        if isinstance(value,dict):return {k:rebind(v) for k,v in value.items()}
        if isinstance(value,list):return [rebind(v) for v in value]
        if isinstance(value,str) and value.startswith(old+'/'):return new+value[len(old):]
        return value
    cfg=rebind(cfg)
    cfg['garmi_task']['kitchen']['files_sha256']['mask']=sha(target/'position-mask.json')
    (target/'task.yaml').write_text(yaml.safe_dump(cfg))
    prep=json.loads((target/'preparation.json').read_text())
    prep.update(status='published_inputs_pending_runtime_path_check',
                source_configuration_sha256=sha(source/'task.yaml'))
    prep.pop('runtime_reset_check_sha256',None)
    save(target/'preparation.json',prep)
    save(target/'provenance.json',dict(status='40_locally_validated_references_expert_approach_pending',
        source_directory=str(source.relative_to(ROOT)),source_configuration_sha256=sha(source/'task.yaml'),
        source_runtime_check_sha256=sha(source/'reset-check/summary.json'),
        published_configuration_sha256=sha(target/'task.yaml'),reference_count=40,
        expert_trajectories_generated=0,optimizer_updates=0,
        note='Runtime input paths use project resources and assets. Full historical physics evidence remains under runs; compact accepted evidence is copied here.'))
    print('PUBLISHED_FOR_RESET_CHECK',target,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('object',choices=['mug','pitcher'])
    p.add_argument('--root',type=Path);p.add_argument('--destination',type=Path);args=p.parse_args()
    main(args.object,args.root,args.destination)
