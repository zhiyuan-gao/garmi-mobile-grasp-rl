"""Balance finger closure on GraspGen pitcher poses, then rescreen geometry.

This is deterministic geometric refinement, not new network inference, training,
or physical validation. Keep the exact original pose and displacement provenance.
"""
import json
import os
import numpy as np
import trimesh
from prepare_fresh_pitcher import ASSET, MESH
from prepare_rl_validation import hand_parts, save
from rebalance_mug import FingerContacts
from select_references import ROOT, HAND_OFFSET, screen, sha


def main():
    source=ROOT/'runs/pitcher-fresh-v3/019_pitcher_base'
    out=ROOT/'runs/pitcher-centered-pool-v4/019_pitcher_base'
    out.mkdir(parents=True,exist_ok=False)
    pool=np.load(source/'candidate_pool.npz')
    old=json.loads((source/'candidate-contacts.json').read_text())['records']
    old=[r for r in old if r['eligible']]
    boundary=source/'cooked-boundary.obj'
    cooked=FingerContacts(trimesh.load(boundary,force='mesh',process=False))
    poses=[];indices=[];displacements=[]
    for j,r in enumerate(old):
        i=r['candidate_index'];original=pool['T_object_graspgen'][i];t=original.copy()
        contacts=r['cooked_contacts']
        for _ in range(2):
            if not all(c is not None for c in contacts):break
            shift=(contacts[0]['q_m']-contacts[1]['q_m'])/2
            # Left finger opens along +hand Y = +GraspGen X.
            t[:3,3]+=shift*t[:3,0]
            contacts=cooked.query(t)
        poses.append(t);indices.append(i);displacements.append(float(np.linalg.norm(t[:3,3]-original[:3,3])))
        if j%200==0:print('CENTERED',j,flush=True)
    poses=np.array(poses);indices=np.array(indices);scores=pool['scores'][indices]
    np.savez_compressed(out/'candidate_pool.npz',T_object_graspgen=poses,scores=scores,
        parent_candidate_indices=indices,parent_T_object_graspgen=pool['T_object_graspgen'][indices],
        recenter_displacement_m=displacements)
    (out/'cooked-boundary.obj').symlink_to(os.path.relpath(boundary,out))
    fingerprints=dict(parent_pool_sha256=sha(source/'candidate_pool.npz'),pool_sha256=sha(out/'candidate_pool.npz'),
        mesh_sha256=sha(MESH),cooked_sha256=sha(ASSET/'cooked.pkl'),robot_sha256=sha(ROOT/'assets/kitchen/cooked-model.pkl'))
    hand=trimesh.util.concatenate(hand_parts());hand.apply_transform(HAND_OFFSET)
    valid=np.ones(len(poses),bool);clearances=[];reports={}
    for label,path in [('scan',MESH),('cooked',boundary)]:
        folder=out/label;folder.mkdir()
        v,c,n=screen(trimesh.load(path,force='mesh',process=False),hand,poses,scores,folder,fingerprints)
        valid &= v;clearances.append(c);reports[label]=n
    np.savez_compressed(out/'geometry-screen.npz',valid=valid,plane_clearance_m=np.minimum(*clearances))
    save(out/'geometry-screen.json',dict(sources=fingerprints,counts=reports))
    scan=FingerContacts(trimesh.load(MESH,force='mesh',process=False))
    direction=np.array([.071,.069,0.]);direction/=np.linalg.norm(direction)
    split=np.array([-.015,.03,0.])+direction*.068
    records=[]
    for j,i in enumerate(np.flatnonzero(valid)):
        a,b=scan.query(poses[i]),cooked.query(poses[i]);both=all(c is not None for c in a+b)
        delta=[abs(x['q_m']-y['q_m']) for x,y in zip(a,b)] if both else []
        imbalance=abs(b[0]['q_m']-b[1]['q_m']) if both else None
        records.append(dict(candidate_index=int(i),parent_candidate_index=int(indices[i]),
            scan_contacts=a,cooked_contacts=b,both_contacts=both,
            maximum_contact_travel_difference_m=max(delta) if delta else None,
            closure_imbalance_m=imbalance,recenter_displacement_m=displacements[i],
            eligible=bool(both and max(delta)<=.002 and imbalance<=.001),
            handle=bool(both and all((np.array(c['point_m'])-split)@direction>0 for c in a+b))))
        if j%200==0:print('CONTACT',j,flush=True)
    save(out/'candidate-contacts.json',dict(sources=fingerprints,records=records))
    save(out.parent/'provenance.json',dict(method='Two geometric translations along closing axis to balance independent first-contact joint positions; orientation unchanged.',
        sources=fingerprints,parent_graspgen_inference_reused=True,new_network_inference=False,
        count=len(poses),eligible=sum(r['eligible'] for r in records),maximum_translation_m=max(displacements),
        physical_validation='pending',expert_trajectories_generated=0,optimizer_updates=0))
    print('CENTERED_POOL_READY',sum(r['eligible'] for r in records),flush=True)


if __name__=='__main__':main()
