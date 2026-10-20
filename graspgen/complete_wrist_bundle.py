"""Mark a packaged bundle complete only after its real runtime check and previews."""
import argparse,csv,shutil
from select_references import ROOT,sha
from try_handle_roll import read
from prepare_rl_validation import save
from review_wrist_v6 import strict_approach,matching_condition


def main(name):
    folder=ROOT/'resources/validated-ycb-v6'/name
    report=read(folder/'validation.json');check=read(folder/'runtime-check/summary.json')
    assert check['status']=='passed' and check['reference_count']==40
    assert check['configuration_sha256']==report['configuration_sha256']==sha(folder/'task.yaml')
    refs=read(folder/'references.json');assert len(refs)==len({r['id'] for r in refs})==40
    assert report['coverage']['minimum_reference_separation']>1e-4
    for p,digest in read(folder/'position-mask.json')['sources_sha256'].items():assert sha(ROOT/p)==digest
    for path,digest in report['runtime_code_sha256'].items():assert sha(ROOT/path)==digest
    for c in report['references']:
        a=c['approach_evidence'];r=a['result'];assert strict_approach(r)
        assert sha(ROOT/a['saved_trace'])==r['sha256']
        assert sha(ROOT/a['summary'])==a['summary_sha256']
        proofs=[a]+c.get('additional_approach_evidence',[])
        for proof in proofs:
            assert strict_approach(proof['result'])
            assert sha(ROOT/proof['saved_trace'])==proof['result']['sha256']
            assert sha(ROOT/proof['summary'])==proof['summary_sha256']
            assert sha(ROOT/proof['runtime_code']['snapshot'])==proof['runtime_code']['snapshot_sha256']
        assert all(any(matching_condition(condition,proof['result']) for proof in proofs) for condition in c['conditions'])
        for e in c['endpoint_evidence']:
            assert e['condition']['passed'] and sha(ROOT/e['folder']/'review.json')==e['review_sha256']
    for tag in ['front','reverse']:assert (folder/f'40-grasps-{tag}.png').stat().st_size>10000
    assert len(read(folder/'preview-manifest.json')['rows'])==40
    report.update(status='completed_40_distinct_strictly_verified_references',
        runtime_check_sha256=sha(folder/'runtime-check/summary.json'),
        runtime_check_script_sha256=sha(ROOT/'rl/check_wrist_reference_bundle.py'))
    save(folder/'validation.json',report)
    with (folder/'reference-index.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['number','reference','handle','inner_outer','opening_m','lift_m','slip_m','hold_seconds','validation_trace'])
        for i,(ref,c) in enumerate(zip(refs,report['references']),1):
            r=c['approach_evidence']['result'];assert ref['id']==c['ref']['id']
            w.writerow([i,ref['id'],c['handle'],'innerouter' in ref['id'],ref['opening_m'],r['height'],r['slip'],r['hold_seconds'],c['approach_evidence']['saved_trace']])
    alias=ROOT/f'configs/{name}-wrist-v6.yaml';shutil.copy2(folder/'task.yaml',alias)
    assert sha(alias)==sha(folder/'task.yaml')
    save(folder/'summary.json',dict(status='complete',object=name,reference_count=40,
        handle_count=report['handle_count'],inner_outer_handle_count=report['inner_outer_handle_count'],
        contact_policy=report['contact_policy'],configuration=str(alias.relative_to(ROOT)),configuration_sha256=sha(alias),
        validation_sha256=sha(folder/'validation.json'),scope=report['scope'],training_started=False,
        saved_traces_purpose='Validation evidence only; not a formal expert training dataset'))
    save(folder/'SHA256.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='SHA256.json'})
    print('COMPLETED',name,40,'handle',report['handle_count'],'inner_outer',report['inner_outer_handle_count'])


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('object',choices=['mug','pitcher']);a=p.parse_args();main(a.object)
