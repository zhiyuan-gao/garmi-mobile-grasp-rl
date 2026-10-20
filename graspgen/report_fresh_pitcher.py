"""Static images and an auditable summary of fresh pitcher grasp diagnostics."""
import argparse
import csv
import json

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from prepare_fresh_pitcher import OUT, ASSET, MESH, BOUNDARY, fingerprints
from prepare_rl_validation import hand_parts, save
from select_references import ROOT, HAND_OFFSET, sha, pose_components


def preview():
    data=np.load(OUT/'selected-40.npz');poses=data['T_object_graspgen'];scores=data['scores']
    refs=json.loads((OUT/'selected-40.json').read_text())
    selection=json.loads((OUT/'selection-report.json').read_text())
    mesh=trimesh.load(MESH,force='mesh',process=False)
    hand=trimesh.util.concatenate(hand_parts());hand.apply_transform(HAND_OFFSET)
    pc,_=trimesh.sample.sample_surface(mesh,3000,seed=42)
    physics=json.loads((OUT/'physics/summary.json').read_text()) if (OUT/'physics/summary.json').exists() else None
    fig=plt.figure(figsize=(20,15),facecolor='white')
    for j,g in enumerate(poses):
        ax=fig.add_subplot(5,8,j+1,projection='3d');hv=trimesh.transform_points(hand.vertices,g)
        ax.scatter(*pc.T,s=.35,c='#188e78',alpha=.4,depthshade=False)
        ax.add_collection3d(Poly3DCollection(hv[hand.faces],facecolors='#e6af31',linewidths=0,alpha=.8))
        v=np.vstack([mesh.vertices,hv]);c=(v.min(0)+v.max(0))/2;r=np.ptp(v,axis=0).max()*.54
        ax.set_xlim(c[0]-r,c[0]+r);ax.set_ylim(c[1]-r,c[1]+r);ax.set_zlim(c[2]-r,c[2]+r)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-60);ax.set_axis_off()
        kind='handle' if selection['references'][j]['handle'] else 'other'
        label=f'{j+1:02d} | {kind} | {scores[j]:.2f}';color='#333333'
        if physics:
            result=physics['per_reference'][refs[j]['id']]
            label+=f"\nLift tests: {result['successes']}/{result['attempts']}"
            color='#17734b' if result['successes'] else '#b03c32'
        ax.set_title(label,fontsize=9,y=.95,color=color)
    fig.suptitle('Pitcher | 40 NEW references (pitcher-v3-ref001...040) | yellow: actual open hand/wrist colliders',fontsize=16)
    fig.text(.5,.01,'Original scan sampled anew; corrected contact-v3-coarse physics. Direct IK diagnostics; no expert trajectories or training.',ha='center',fontsize=11)
    fig.subplots_adjust(left=0,right=1,bottom=.035,top=.94,wspace=-.08,hspace=.08)
    fig.savefig(OUT/'gallery-40.png',dpi=160);plt.close(fig)

    cloud=np.load(OUT/'input-pointclouds.npz')['points'][0]
    centers,rotations=pose_components(poses)
    fig=plt.figure(figsize=(15,6),facecolor='white')
    for j in range(2):
        ax=fig.add_subplot(1,2,j+1,projection='3d')
        points=cloud if j==0 else pc
        ax.scatter(*points.T,s=2 if j==0 else .5,c='#188e78',alpha=.6,depthshade=False)
        if j:
            ax.scatter(*centers.T,c=np.arange(40),cmap='turbo',s=30)
            for i,c in enumerate(centers):
                ax.quiver(*c,*(-rotations[i,:,2]*.035),color='#ba7211',arrow_length_ratio=.25)
                ax.text(*c,str(i+1),fontsize=7)
        v=np.vstack([mesh.vertices,centers]);c=(v.min(0)+v.max(0))/2;r=np.ptp(v,axis=0).max()*.55
        ax.set_xlim(c[0]-r,c[0]+r);ax.set_ylim(c[1]-r,c[1]+r);ax.set_zlim(c[2]-r,c[2]+r)
        ax.set_box_aspect((1,1,1));ax.view_init(23,-60)
        ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
        ax.set_title('Actual first GraspGen input: 2048 sampled points' if j==0 else '40 selected TCPs and outward approach directions')
    fig.suptitle(f"Fresh pitcher candidates: {selection['total_candidates']} | qualified: {selection['contact_eligible']} | handle references: {selection['handle_reference_count']}/40",fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'coverage-40.png',dpi=160);plt.close(fig)
    print('IMAGES',OUT/'gallery-40.png',OUT/'coverage-40.png',flush=True)


def summarize():
    physics=json.loads((OUT/'physics/summary.json').read_text())
    assert physics['status']=='completed' and physics['configuration_sha256']==sha(OUT/'task.yaml')
    assert physics['additional_object_collision_mesh_sha256']==sha(BOUNDARY)
    selection=json.loads((OUT/'selection-report.json').read_text())
    assert selection['sources']==fingerprints()
    refs=json.loads((OUT/'selected-40.json').read_text())
    assert set(physics['per_reference'])=={r['id'] for r in refs}
    provenance=json.loads((OUT.parent/'pool-provenance.json').read_text())
    assert provenance['objects']['019_pitcher_base']['original_count']==0
    assert physics['expert_trajectories_generated']==physics['optimizer_updates']==0
    preserved_inputs=json.loads((OUT.parent/'preserved-inputs.json').read_text())
    assert all(sha(ROOT/path)==digest for path,digest in preserved_inputs.items())
    preserved={}
    for y,name in [('025_mug','mug'),('019_pitcher_base','pitcher')]:
        old=json.loads((ROOT/'runs/ycb-rl-validation'/y/'bank.json').read_text())['objects'][0]
        preserved[name]=dict(original_usd_unchanged=sha(ROOT/'assets/ycb'/y/'rl.usda')==old['usd_sha256'],
            published_references_unchanged=json.loads((ROOT/'resources'/f'{name}-40-references.json').read_text())==old['references'])
        assert all(preserved[name].values())
    with (OUT.parent/'reference-validation.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=['reference','candidate_index','score','handle','max_contact_travel_difference_mm','placement_cells','attempts','successes','reasons'])
        writer.writeheader()
        mask=json.loads((OUT/'position-mask.json').read_text())
        for j,ref in enumerate(refs):
            s=selection['references'][j];p=physics['per_reference'][ref['id']]
            writer.writerow(dict(reference=ref['id'],candidate_index=s['candidate_index'],score=ref['score'],handle=s['handle'],
                max_contact_travel_difference_mm=s['maximum_contact_travel_difference_m']*1000,
                placement_cells=mask['allowed_cells_per_reference'][j],attempts=p['attempts'],successes=p['successes'],reasons=json.dumps(p['reasons'])))
    handle=[s['reference'] for s in selection['references'] if s['handle']]
    old=json.loads((ROOT/'runs/pitcher-handle-audit/coarse/physics/summary.json').read_text())
    report=dict(status='fresh_pitcher_references_diagnosed_expert_pending_confirmation',
        collision_asset=str(ASSET.relative_to(ROOT)),task_configuration=str((OUT/'task.yaml').relative_to(ROOT)),
        fresh_candidates_only=True,candidate_count=selection['total_candidates'],reference_count=40,
        geometry_eligible=selection['geometry_eligible'],contact_eligible=selection['contact_eligible'],
        handle_reference_count=len(handle),coverage=selection['coverage'],
        maximum_selected_contact_travel_difference_m=selection['maximum_selected_contact_travel_difference_m'],
        physical_case_count=physics['case_count'],references_with_success=physics['references_with_success'],physical_counts=physics['counts'],
        handle_references_with_success=sum(physics['per_reference'][r]['successes']>0 for r in handle),
        reset=json.loads((OUT/'physics/reset-check.json').read_text()),
        input_and_placement_audit=json.loads((OUT.parent/'input-and-placement-audit.json').read_text()),
        old_references_same_collision_asset=dict(cases=old['case_count'],counts=old['counts'],references_with_success=old['references_with_success']),
        published_assets_preserved=preserved,preserved_input_sha256=preserved_inputs,default_configuration_changed=False,
        provenance=provenance,sources=fingerprints(),expert_trajectories_generated=0,optimizer_updates=0,
        limitations=['Fresh full-surface scan samples, not an occluded camera point cloud.',
            'Pitcher body remains solid; no interior reconstruction.',
            'Coverage optimized over the finite qualified pool, not all possible grasps.',
            'Finite IK seeds and placements; single run, no statistical success-rate estimate.',
            'Static endpoint checks now include the corrected cooked boundary; old-reference counts are contextual, not a controlled comparison.',
            'Direct IK initialization only; expert approach trajectories require confirmation.'])
    selection['status']='rl_diagnostics_completed_expert_pending_confirmation'
    save(OUT/'selection-report.json',selection);save(OUT.parent/'validation.json',report)
    print(json.dumps({k:report[k] for k in ['candidate_count','contact_eligible','handle_reference_count','physical_case_count','references_with_success','physical_counts','handle_references_with_success']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('step',choices=['preview','summarize'])
    args=parser.parse_args();preview() if args.step=='preview' else summarize()
