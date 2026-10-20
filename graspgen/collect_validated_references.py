"""Collect verified repeatability evidence; count distinct poses, not placements."""
import json
import numpy as np
from select_references import ROOT, sha, pose_components, distances_to
from prepare_rl_validation import save

OUT=ROOT/'runs/validated-reference-bank-v4'


def main():
    overview={}
    for name in ('mug','pitcher'):
        reviews=([ROOT/'runs/grasp-repeatability-v3/mug/review.json'] if name=='mug' else
                 [ROOT/'runs/pitcher-supplement-v3/narrow-repeatability/pitcher/review.json'])
        reviews+=sorted((ROOT/'runs').glob(f'grasp-expansion-v4*/repeat-{name}/{name}/review.json'))
        reviews+=sorted((ROOT/'runs').glob(f'grasp-expansion-v4*/repeat-{name}-extra*/{name}/review.json'))
        allrefs=[];evidence=[];seen={}
        for path in reviews:
            if not path.exists():continue
            report=json.loads(path.read_text());folder=path.parent
            if report['retained_reference_count']==0:
                assert not report['retained_reference_ids']
                continue
            inputs=json.loads((folder/'inputs.json').read_text())
            assert report['sources_sha256']['inputs']==sha(folder/'inputs.json')
            assert report['sources_sha256']['physics']==sha(folder/'physics/summary.json')
            assert inputs['criteria_sha256']==sha(folder.parent/'criteria.json')
            source=ROOT/inputs['source']
            assert report['sources_sha256']['source_bank']==sha(source/'bank.json')
            bank=json.loads((source/'bank.json').read_text())
            refs={r['id']:r for r in bank['objects'][0]['references']}
            radius=max(abs(v) for pair in report['criteria']['object_offsets_m'] for v in pair)
            for identifier in report['retained_reference_ids']:
                conditions=[c for c in report['conditions'] if c['reference']==identifier and c['passed']]
                assert conditions
                for c in conditions:
                    n=c['successes'];rule=report['criteria']
                    assert n['nominal']>=rule['nominal_minimum_successes'] and n['object_offset']>=rule['object_minimum_successes'] and n['base_offset']>=rule['base_minimum_successes']
                row=dict(reference=identifier,handle=conditions[0]['handle'],
                    object_offset_radius_m=radius,conditions=conditions,source=str(source.relative_to(ROOT)),
                    review=str(path.relative_to(ROOT)),review_sha256=sha(path),criteria=report['criteria'])
                if identifier in seen:
                    i=seen[identifier]
                    assert np.allclose(allrefs[i]['mesh_T_gripper'],refs[identifier]['mesh_T_gripper'],atol=1e-10)
                    if radius>evidence[i]['object_offset_radius_m']:
                        row['additional_reviews']=[evidence[i]];evidence[i]=row
                    else:evidence[i].setdefault('additional_reviews',[]).append(row)
                    continue
                seen[identifier]=len(allrefs)
                allrefs.append(refs[identifier]);evidence.append(row)
        # Cross-pool exact duplicates, including exchanged parallel fingers, are
        # not counted twice. Keep every evidence file even when an entry merges.
        keep=[];duplicates=[]
        if allrefs:
            poses=np.array([r['mesh_T_gripper'] for r in allrefs]);c,r=pose_components(poses)
            for i in range(len(allrefs)):
                d=distances_to(c,r,i)
                match=next((j for j in keep if d[j]<1e-4),None)
                if match is None:keep.append(i)
                else:duplicates.append(dict(reference=allrefs[i]['id'],same_pose_as=allrefs[match]['id']))
        refs=[allrefs[i] for i in keep];rows=[evidence[i] for i in keep]
        save(OUT/name/'all-validated-references.json',refs)
        save(OUT/name/'validation-evidence.json',dict(references=rows,exact_duplicate_poses=duplicates))
        broad=[r for r in rows if r['object_offset_radius_m']>=.005-1e-9]
        overview[name]=dict(distinct_validated_references=len(refs),handle_references=sum(r['handle'] for r in rows),
            validated_at_5mm=len(broad),validated_only_at_2_5mm=len(rows)-len(broad),
            target_reference_count=40,remaining_to_40=max(0,40-len(refs)),
            reference_file=str((OUT/name/'all-validated-references.json').relative_to(ROOT)),
            evidence_file=str((OUT/name/'validation-evidence.json').relative_to(ROOT)))
        final=OUT/'final'/name/'expert-inputs'
        if (final/'task.yaml').exists():
            selected=json.loads((final/'bank.json').read_text())['objects'][0]['references']
            overview[name].update(selected_reference_count=len(selected),
                selected_task=str((final/'task.yaml').relative_to(ROOT)),runtime_inputs_checked=False)
            check=final/'reset-check/summary.json'
            if check.exists():
                runtime=json.loads(check.read_text())
                assert runtime['status']=='passed' and runtime['configuration_sha256']==sha(final/'task.yaml')
                overview[name]['runtime_inputs_checked']=True
        published=ROOT/'resources/validated-ycb-v4'/name
        if (published/'task.yaml').exists():
            overview[name].update(published_reference_file=str((published/'references.json').relative_to(ROOT)),
                published_task=str((published/'task.yaml').relative_to(ROOT)),published_runtime_checked=False)
            check=published/'reset-check/summary.json'
            if check.exists():
                runtime=json.loads(check.read_text())
                assert runtime['status']=='passed' and runtime['configuration_sha256']==sha(published/'task.yaml')
                overview[name]['published_runtime_checked']=True
    complete=all(x['distinct_validated_references']>=40 for x in overview.values())
    finalized=all(x.get('selected_reference_count')==40 and x.get('runtime_inputs_checked')
                  and x.get('published_runtime_checked') for x in overview.values())
    save(OUT/'summary.json',dict(status=('forty_each_local_validation_complete_expert_confirmation_pending' if finalized else
        'enough_validated_candidates_final_selection_pending' if complete else 'incomplete_continue_validation'),
        objects=overview,expert_trajectories_generated=0,optimizer_updates=0,
        limitation='Local physical grasp validation only; expert approach trajectories have not been generated or tested.'))
    print(json.dumps(overview,indent=2))


if __name__=='__main__':main()
