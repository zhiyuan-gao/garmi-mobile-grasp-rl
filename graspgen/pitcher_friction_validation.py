"""Freeze and review v5 pitcher revalidation; never generate expert motion."""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import yaml
from select_references import ROOT,sha,pose_components,distances_to,coverage
from prepare_rl_validation import save
from prepare_grasp_repeatability import CRITERIA

OUT=ROOT/'runs/validated-reference-bank-v5'
ASSET=ROOT/'assets/ycb/019_pitcher_base/contact-v5-friction'

def inside(p):return '/workspace/'+str(p.relative_to(ROOT))
def read(p):return json.loads(p.read_text())

def old_rows():
    prior=read(ROOT/'runs/validated-reference-bank-v4/pitcher/validation-evidence.json')['references']
    old_ids={r['reference'] for r in prior}
    paths={ROOT/r['review'] for r in prior}
    paths.update((ROOT/'runs').glob('grasp-expansion-v4*/repeat-pitcher*/pitcher/review.json'))
    paths.update([ROOT/'runs/grasp-repeatability-v3/pitcher/review.json',
                  ROOT/'runs/pitcher-supplement-v3/repeatability/pitcher/review.json'])
    best={}
    for path in sorted(paths):
        if not path.exists():continue
        report=read(path)
        if 'sources_sha256' not in report:continue
        assert report['sources_sha256']['physics']==sha(path.parent/'physics/summary.json')
        assert report['sources_sha256']['inputs']==sha(path.parent/'inputs.json')
        source=ROOT/read(path.parent/'inputs.json')['source']
        assert report['sources_sha256']['source_bank']==sha(source/'bank.json')
        obj=read(source/'bank.json')['objects'][0]
        refs={r['id']:r for r in obj['references']}
        for c in report['conditions']:
            identifier=c['reference'];s=c['successes']
            quality=(bool(c['passed']),s['object_offset'],sum(s.values()),s['nominal'])
            row=dict(reference=identifier,ref=refs[identifier],handle=c['handle'],
                condition={k:copy.deepcopy(c[k]) for k in ['id','reference','region','depth','object_state','base_pose','arm_position','handle']},
                source=str(source.relative_to(ROOT)),old_review=str(path.relative_to(ROOT)),
                old_review_sha256=sha(path),old_successes=s,old_retained=identifier in old_ids,
                old_object_radius=max(abs(x) for pair in report['criteria']['object_offsets_m'] for x in pair),quality=quality)
            if identifier not in best or quality>best[identifier]['quality']:best[identifier]=row
    rows=sorted(best.values(),key=lambda r:(r['old_retained'],r['quality']),reverse=True)
    assert sum(r['old_retained'] for r in rows)==28
    # Exchanged-finger exact duplicates never become separate references.
    poses=np.array([r['ref']['mesh_T_gripper'] for r in rows]);c,q=pose_components(poses)
    keep=[]
    for i in range(len(rows)):
        distances=distances_to(c,q,i)
        if not any(distances[j]<1e-4 for j in keep):keep.append(i)
    return [rows[i] for i in keep]

def inputs(rows,source):
    source.mkdir(parents=True,exist_ok=False)
    obj=read(ROOT/rows[0]['source']/'bank.json')['objects'][0]
    geometry={k:v for k,v in obj.items() if k not in ['references','usd','usd_sha256','friction_assumption','cooked','cooked_sha256']}
    for row in rows:
        other=read(ROOT/row['source']/'bank.json')['objects'][0]
        assert {k:v for k,v in other.items() if k in geometry}==geometry
    obj.update(references=[r['ref'] for r in rows],usd=inside(ASSET/'rl.usda'),usd_sha256=sha(ASSET/'rl.usda'),friction_assumption=2.)
    save(source/'bank.json',dict(status='v5_reference_physics_pending',objects=[obj]))
    ids=[r['reference'] for r in rows];groups={};checked_hashes={}
    for i,row in enumerate(rows):
        c=row['condition'];xy=np.array([.3-c['object_state'][1],6.35+c['object_state'][0]])
        key=tuple(np.round(xy,6));lo=np.array(key)-.005;hi=np.array(key)+.005
        oldmask=read(ROOT/row['source']/'position-mask.json');idx=oldmask['reference_ids'].index(row['reference'])
        for p,h in oldmask['sources_sha256'].items():
            if p not in checked_hashes:checked_hashes[p]=sha(ROOT/p)
            assert checked_hashes[p]==h
        assert any((lo>=np.array(cell['bounds_map_xy'][0])-1e-7).all() and
            (hi<=np.array(cell['bounds_map_xy'][1])+1e-7).all() and idx in cell['allowed_reference_indices']
            for cell in oldmask['cells']),row['reference']
        groups.setdefault(key,[]).append(i)
    cells=[];regions=[];settings=[]
    for j,(xy,indices) in enumerate(sorted(groups.items())):
        bounds=[(np.array(xy)-.005).tolist(),(np.array(xy)+.005).tolist()];name=f'v5-local-{j:02d}'
        cells.append(dict(region=name,bounds_map_xy=bounds,allowed_reference_indices=indices))
        regions.append(dict(name=name,bounds_map_xy=bounds,shape_xy=[1,1],cell_offset=j))
        settings.append(dict(name=name,bounds_map_xy=bounds,probability=1/len(groups)))
    for i,a in enumerate(cells):
        for b in cells[i+1:]:
            assert not (np.minimum(a['bounds_map_xy'][1],b['bounds_map_xy'][1])-np.maximum(a['bounds_map_xy'][0],b['bounds_map_xy'][0])>1e-8).all()
    mask=read(ROOT/rows[0]['source']/'position-mask.json')
    mask['sources_sha256'].pop(mask['bank_path'])
    mask.update(bank_path=str((source/'bank.json').relative_to(ROOT)),reference_ids=ids,regions=regions,cells=cells,
        cell_size_m=.01,empty_cells=[],allowed_cells_per_reference=[1]*len(ids),status='validated',
        method='Reindexed +/-5 mm local windows, each wholly inside its prior geometry-checked cell. Only material changed.',
        scope='Static geometry checked; physical success under v5 friction must be established separately.')
    mask['sources_sha256'][mask['bank_path']]=sha(source/'bank.json')
    save(source/'position-mask.json',mask)
    cfg=yaml.safe_load((ROOT/'configs/pitcher-friction-v5.yaml').read_text());task=cfg['garmi_task']
    task['multi_object'].update(manifest=inside(source/'bank.json'),reference_ids={'pitcher':ids})
    task['kitchen'].update(mask=inside(source/'position-mask.json'),bowl_regions=settings,sampling_mode='validated_subregions')
    task['kitchen']['files_sha256']['mask']=sha(source/'position-mask.json')
    (source/'task.yaml').write_text(yaml.safe_dump(cfg))
    save(source/'provenance.json',dict(rows=rows,geometry_unchanged=True,friction=2.,old_success_labels_not_inherited=True))
    return source

def prepare():
    assert not OUT.exists()
    rows=old_rows();save(OUT/'candidate-seeds.json',rows)
    old=[r for r in rows if r['old_retained']];other=[r for r in rows if not r['old_retained']]
    batches=[('a',old),('b',other[:60]),('c',other[60:120])]
    save(OUT/'unselected-seeds.json',other[120:])
    for name,selected in batches:
        if selected:freeze(name,selected)
    print('PREPARED',len(old),'old retained',len(other),'additional candidates',flush=True)

def freeze(name,rows):
    folder=OUT/f'batch-{name}';source=inputs(rows,folder/'source')
    target=folder/'pitcher';target.mkdir();save(folder/'criteria.json',CRITERIA)
    conditions=[r['condition'] for r in rows];cases=[]
    variants=[('nominal',[0.,0.])]*5+[('object_offset',x) for x in CRITERIA['object_offsets_m']]+[('base_offset',x) for x in CRITERIA['base_offsets_m']]
    for c in conditions:
        for i,(kind,offset) in enumerate(variants):
            case=dict(id=f"{c['id']}-{i:02d}",condition_id=c['id'],object='pitcher',reference=c['reference'],
                region=c['region'],depth=c['depth'],object_state=c['object_state'].copy(),
                robot_state=[0.,.05,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
                endpoint_base_pose=c['base_pose'].copy(),endpoint_arm_seed=c['arm_position'].copy(),
                test_kind=kind,offset_task_xy_m=offset)
            if kind=='object_offset':case['object_state'][:2]=(np.array(case['object_state'][:2])+offset).tolist()
            if kind=='base_offset':case['endpoint_base_pose'][:2]=(np.array(case['endpoint_base_pose'][:2])+offset).tolist()
            cases.append(case)
    random.Random(2026101901).shuffle(cases)
    save(target/'cases.json',cases);save(target/'conditions.json',conditions)
    import shutil
    shutil.copy2(ROOT/'runs/pitcher-supplement-v3/019_pitcher_base/cooked-boundary.obj',target/'cooked-boundary.obj')
    save(target/'inputs.json',dict(source=str(source.relative_to(ROOT)),asset=str(ASSET.relative_to(ROOT)),
        condition_count=len(conditions),case_count=len(cases),reference_count=len(rows),
        criteria_sha256=sha(folder/'criteria.json'),cases_sha256=sha(target/'cases.json'),
        cooked_sha256=sha(ASSET/'cooked.pkl'),cooked_boundary_sha256=sha(target/'cooked-boundary.obj'),
        configuration_sha256=sha(source/'task.yaml'),expert_trajectories_generated=0,optimizer_updates=0))
    print('FROZEN',name,len(rows),'references',len(cases),'cases',flush=True)

def review(batch):
    import review_grasp_repeatability as module
    module.OUT=OUT/f'batch-{batch}';module.SOURCES={'pitcher':module.OUT/'source'};module.ASSETS={'pitcher':ASSET}
    physics=read(module.OUT/'pitcher/physics/summary.json')
    assert not physics.get('diagnostic_overrides')
    module.review('pitcher')

def prepare_extra():
    excluded={r['reference'] for r in read(OUT/'candidate-seeds.json')};best={}
    for letter in ['a','c','e','f','g']:
        source=ROOT/f'runs/grasp-expansion-v4{letter}/pitcher'
        physics=read(source/'physics/summary.json')
        assert physics['configuration_sha256']==sha(source/'task.yaml')
        originals={(c['reference'],c['region'],c['depth']):c for c in read(source/'physics/cases.json')}
        refs={r['id']:r for r in read(source/'bank.json')['objects'][0]['references']}
        handles={r['reference']:r['handle'] for r in read(source/'selection-report.json')['references']}
        for r in physics['results']:
            identifier=r['reference']
            if identifier in excluded or r['reason']!='lift_or_hold_failed':continue
            original=originals[(identifier,r['region'],r['depth'])]
            quality=(r['max_hold_s'],r['max_lift_m'],refs[identifier]['score'])
            c=dict(id=f"{identifier}-{r['region']}-{r['depth']}",reference=identifier,region=r['region'],depth=r['depth'],
                object_state=original['object_state'],base_pose=r['base_pose'],arm_position=r['arm_position'],handle=handles[identifier])
            row=dict(reference=identifier,ref=refs[identifier],handle=handles[identifier],condition=c,
                source=str(source.relative_to(ROOT)),old_physics=str((source/'physics/summary.json').relative_to(ROOT)),
                old_physics_sha256=sha(source/'physics/summary.json'),quality=quality)
            if identifier not in best or quality>best[identifier]['quality']:best[identifier]=row
    rows=sorted(best.values(),key=lambda r:r['quality'],reverse=True)[:120]
    folder=OUT/'initial-d';source=inputs(rows,folder/'source');cases=[]
    for row in rows:
        c=row['condition']
        cases.append(dict(id=c['id'],condition_id=c['id'],object='pitcher',reference=c['reference'],region=c['region'],depth=c['depth'],
            object_state=c['object_state'],robot_state=[0.,.05,.0259,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            endpoint_base_pose=c['base_pose'],endpoint_arm_seed=c['arm_position'],test_kind='nominal'))
    save(folder/'cases.json',cases);save(folder/'seeds.json',rows)
    print('ADDITIONAL_INITIAL',len(rows),flush=True)

def freeze_extra():
    folder=OUT/'initial-d';physics=read(folder/'physics/summary.json')
    assert physics['configuration_sha256']==sha(folder/'source/task.yaml')
    assert physics['prescribed_cases_sha256']==sha(folder/'cases.json')
    assert not physics.get('diagnostic_overrides')
    passed={r['reference']:r for r in physics['results'] if r['passed']};rows=[]
    for row in read(folder/'seeds.json'):
        if row['reference'] not in passed:continue
        r=passed[row['reference']]
        row['condition'].update(base_pose=r['base_pose'],arm_position=r['arm_position'])
        row.update(v5_initial_physics_sha256=sha(folder/'physics/summary.json'),v5_initial_case_id=r['id'])
        rows.append(row)
    assert rows
    freeze('d',rows)

def collect():
    refs=[];rows=[];seen=set()
    for path in sorted(OUT.glob('batch-*/pitcher/review.json')):
        report=read(path);inp=read(path.parent/'inputs.json');source=ROOT/inp['source']
        assert report['criteria']==CRITERIA
        assert report['sources_sha256']['source_usd']==sha(ASSET/'rl.usda')
        assert report['sources_sha256']['physics']==sha(path.parent/'physics/summary.json')
        assert report['sources_sha256']['source_bank']==sha(source/'bank.json')
        source_refs={r['id']:r for r in read(source/'bank.json')['objects'][0]['references']}
        for identifier in report['retained_reference_ids']:
            if identifier in seen:continue
            seen.add(identifier);conditions=[c for c in report['conditions'] if c['passed'] and c['reference']==identifier]
            refs.append(source_refs[identifier]);rows.append(dict(reference=identifier,handle=conditions[0]['handle'],
                source=str(source.relative_to(ROOT)),review=str(path.relative_to(ROOT)),review_sha256=sha(path),
                criteria=CRITERIA,object_offset_radius_m=.005,conditions=conditions))
    keep=[]
    if refs:
        c,q=pose_components(np.array([r['mesh_T_gripper'] for r in refs]))
        for i in range(len(refs)):
            distances=distances_to(c,q,i)
            if not any(distances[j]<1e-4 for j in keep):keep.append(i)
    refs=[refs[i] for i in keep];rows=[rows[i] for i in keep]
    save(OUT/'pitcher/all-validated-references.json',refs);save(OUT/'pitcher/validation-evidence.json',dict(references=rows))
    report=dict(status='enough_candidates_selection_pending' if len(refs)>=40 else 'incomplete_continue_validation',
        validated_references=len(refs),handle_references=sum(r['handle'] for r in rows),target=40,
        friction=2.,object_offset_radius_m=.005,expert_trajectories_generated=0,optimizer_updates=0)
    final=OUT/'final/pitcher/expert-inputs'
    published=ROOT/'resources/validated-ycb-v5/pitcher'
    if (published/'reset-check/summary.json').exists():
        for folder in [final,published]:
            check=read(folder/'reset-check/summary.json')
            assert check['status']=='passed' and check['configuration_sha256']==sha(folder/'task.yaml')
        selected=read(published/'references.json')
        assert len(selected)==40 and {r['id'] for r in selected}<={r['id'] for r in refs}
        report.update(status='40_references_ready_expert_approach_requires_confirmation',selected_reference_count=40,
            selected_handle_count=sum(r['handle'] for r in rows if r['reference'] in {x['id'] for x in selected}),
            published_configuration=str((published/'task.yaml').relative_to(ROOT)),runtime_inputs_checked=True)
    save(OUT/'summary.json',report);print(json.dumps(report),flush=True)
    return refs,rows

def finalize():
    refs,rows=collect();assert len(refs)>=40
    poses=np.array([r['mesh_T_gripper'] for r in refs]);c,q=pose_components(poses)
    nearest=np.full(len(refs),np.inf);chosen=[]
    for _ in range(40):
        i=int(np.argmax([r['score'] for r in refs])) if not chosen else int(np.argmax(nearest))
        chosen.append(i);nearest=np.minimum(nearest,distances_to(c,q,i));nearest[chosen]=-np.inf
    selected=[refs[i] for i in chosen];selected_rows=[rows[i] for i in chosen]
    source=inputs([dict(r,ref=ref,condition=r['conditions'][0]) for ref,r in zip(selected,selected_rows)],OUT/'final-source')
    folder=OUT/'final/pitcher';folder.mkdir(parents=True)
    save(folder/'retained-references.json',selected)
    save(folder/'review.json',dict(status='40_distinct_local_repeatability_passed_expert_approach_pending',object='pitcher',
        criteria=CRITERIA,conditions=[r['conditions'][0] for r in selected_rows],provenance=selected_rows,
        retained_reference_count=40,retained_handle_reference_count=sum(r['handle'] for r in selected_rows),
        coverage=coverage(poses,np.array(chosen)),expert_trajectories_generated=0,optimizer_updates=0))
    import prepare_repeatability_expert as module
    module.OUT=OUT/'final';module.SOURCES={'pitcher':source};module.ASSETS={'pitcher':ASSET};module.prepare('pitcher')

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('step',choices=['prepare','review','collect','finalize','prepare_extra','freeze_extra'])
    p.add_argument('--batch');a=p.parse_args()
    if a.step=='review':review(a.batch)
    else:globals()[a.step]()
