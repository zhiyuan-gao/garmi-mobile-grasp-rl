"""Save a partial validated pool with per-reference domains; no runnable task."""
import argparse
import copy
import json
import shutil
from select_references import ROOT,sha
from prepare_grasp_repeatability import ASSETS
from prepare_rl_validation import save


def main(name):
    source=ROOT/'runs/validated-reference-bank-v4'/name
    refs=json.loads((source/'all-validated-references.json').read_text())
    evidence=json.loads((source/'validation-evidence.json').read_text())
    assert refs and len(refs)==len(evidence['references'])
    target=ROOT/'resources/validated-ycb-v4'/f'{name}-progress'
    target.mkdir(parents=True,exist_ok=True)
    rows=evidence['references']
    bank=json.loads((ROOT/rows[0]['source']/'bank.json').read_text())
    obj=bank['objects'][0]
    for key in ['mesh','usd']:assert sha(ROOT/obj[key].removeprefix('/workspace/'))==obj[key+'_sha256']
    for row in rows:
        geometry=json.loads((ROOT/row['source']/'bank.json').read_text())['objects'][0]
        assert all(geometry[key]==obj[key] for key in ['mesh','mesh_sha256','usd','usd_sha256'])
    obj.update(references=refs,cooked='/workspace/'+str((ASSETS[name]/'cooked.pkl').relative_to(ROOT)),
               cooked_sha256=sha(ASSETS[name]/'cooked.pkl'))
    bank['status']='partial_validated_pool_target_40_not_complete_not_a_runnable_task'
    save(target/'bank.json',bank)
    save(target/'references.json',refs)
    save(target/'validation-evidence.json',evidence)
    conditions=[]
    for row in rows:
        assert sha(ROOT/row['review'])==row['review_sha256']
        for c in row['conditions']:
            c=copy.deepcopy(c);c['validated_object_offset_radius_m']=row['object_offset_radius_m'];conditions.append(c)
    save(target/'endpoint-conditions.json',dict(conditions=conditions,
        note='Each reference retains its own tested object/base domain. These endpoints are not expert trajectories or a whole-counter sampling mask.'))
    save(target/'preparation.json',dict(status='partial_pool_target_not_complete',reference_count=len(refs),
        target_reference_count=40,handle_reference_count=sum(r['handle'] for r in rows),
        object_window_halfwidth_m=None,expert_trajectories_generated=0,optimizer_updates=0,
        scope='Evidence archive and preview only. No executable task configuration is supplied for the mixed validation domains.'))
    print('PARTIAL_POOL_ARCHIVED',name,len(refs),target,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('object',choices=['mug','pitcher']);main(p.parse_args().object)
