"""Select exactly 40 broad-domain validated poses and package their conditions.

No expert planner or training is invoked. Refuse to fill a shortage with failed,
untested, or smaller-domain references.
"""
import argparse
import copy
import json
from types import SimpleNamespace
import numpy as np
from select_references import ROOT, sha, pose_components, distances_to, coverage
from prepare_rl_validation import save
from collect_validated_references import OUT


def main(name):
    refs=json.loads((OUT/name/'all-validated-references.json').read_text())
    evidence=json.loads((OUT/name/'validation-evidence.json').read_text())['references']
    keep=[i for i,e in enumerate(evidence) if e['object_offset_radius_m']>=.005-1e-9]
    assert len(keep)>=40, f'{name}: only {len(keep)} references passed the +/-5 mm criteria'
    refs=[refs[i] for i in keep];evidence=[evidence[i] for i in keep]
    poses=np.array([r['mesh_T_gripper'] for r in refs]);centers,rotations=pose_components(poses)
    handle=np.array([e['handle'] for e in evidence]);scores=np.array([r['score'] for r in refs])
    nearest=np.full(len(refs),np.inf);selected=[]
    def choose(allowed):
        allowed=allowed.copy();allowed[selected]=False
        assert allowed.any()
        priority=scores if not selected else nearest
        i=int(np.argmax(np.where(allowed,priority,-np.inf)))
        selected.append(i)
        nearest[:]=np.minimum(nearest,distances_to(centers,rotations,i))
    if name=='mug':
        assert handle.sum()>=10, f'Only {handle.sum()} validated handle grasps; finish requested handle coverage first'
        assert (~handle).sum()>=30
        for _ in range(10):choose(handle)
        for _ in range(30):choose(~handle)
    else:
        for _ in range(40):choose(np.ones(len(refs),bool))
    selected=np.array(selected)
    final_refs=[refs[i] for i in selected];rows=[evidence[i] for i in selected]
    source=OUT/'merged-sources'/name;source.mkdir(parents=True,exist_ok=False)
    bank=json.loads((ROOT/rows[0]['source']/'bank.json').read_text())
    geometry={k:v for k,v in bank['objects'][0].items() if k!='references'}
    for e in rows:
        assert sha(ROOT/e['review'])==e['review_sha256']
        obj=json.loads((ROOT/e['source']/'bank.json').read_text())['objects'][0]
        assert {k:v for k,v in obj.items() if k!='references'}==geometry
    bank['objects'][0]['references']=final_refs
    save(source/'bank.json',bank);save(source/'selected-40.json',final_refs)
    save(source/'selection-report.json',dict(reference_count=40,handle_count=sum(e['handle'] for e in rows),
        method='Farthest pose selection from references passing the same +/-5 mm local validation; mug exactly 10 handle + 30 other.',
        coverage=coverage(poses,selected),references=rows))
    from expand_reference_validation import prepare_geometry
    prepare_geometry(SimpleNamespace(object=name,out=source.parent))
    folder=OUT/'final'/name;folder.mkdir(parents=True,exist_ok=False)
    conditions=[copy.deepcopy(c) for e in rows for c in e['conditions']]
    criteria=rows[0]['criteria']
    assert all(e['criteria']==criteria for e in rows)
    save(folder/'retained-references.json',final_refs)
    save(folder/'review.json',dict(status='40_distinct_local_repeatability_passed_expert_approach_pending',
        object=name,criteria=criteria,conditions=conditions,retained_reference_count=40,
        retained_handle_reference_count=sum(e['handle'] for e in rows),
        provenance=rows,coverage=coverage(poses,selected),expert_trajectories_generated=0,optimizer_updates=0))
    import prepare_repeatability_expert as module
    module.OUT=OUT/'final';module.SOURCES={name:source};module.prepare(name)
    print('FINAL_40_PACKAGED',name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('object',choices=['mug','pitcher'])
    main(p.parse_args().object)
