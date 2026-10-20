"""Bounded local-Z roll search; independent candidates, no learned confidence transfer."""
import argparse, copy, itertools, json, pickle
from collections import Counter
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation
from trimesh.collision import CollisionManager
from select_references import ROOT, HAND_OFFSET, sha, farthest_selection
from prepare_rl_validation import hand_parts, save
from rebalance_mug import FingerContacts
from audit_pitcher_contacts import cooked_boundary

OUT = ROOT/'runs/ycb-handle-roll90-v1'
def host(s): return ROOT/s.removeprefix('/workspace/')
def inside(p): return '/workspace/'+str(p.relative_to(ROOT))
def read(p): return json.loads(p.read_text())

def prepare(name):
    source=ROOT/f'resources/validated-ycb-v{4 if name=="mug" else 5}/{name}'
    out=OUT/name; out.mkdir(parents=True,exist_ok=True)
    bank=read(source/'bank.json');obj=bank['objects'][0]
    gallery=ROOT/f'runs/ycb-expert-precision-v1/{name}/strict-success-preview/manifest.json'
    ids={r['reference'] for r in read(gallery)['rows'] if r['handle']}
    seeds=[r for r in obj['references'] if r['id'] in ids]
    mesh=trimesh.load(host(obj['mesh']),force='mesh',process=False)
    boundary=cooked_boundary(host(obj['cooked']));boundary.export(out/'cooked-boundary.obj')
    contacts=[FingerContacts(m) for m in (mesh,boundary)]
    obstacles=[]
    for m in (mesh,boundary):
        cm=CollisionManager();cm.add_object('object',m);obstacles.append(cm)
    parts=hand_parts();hand=trimesh.util.concatenate(parts);hand.apply_transform(HAND_OFFSET)
    manager=CollisionManager();manager.add_object('hand',hand)
    counts=Counter();records=[];eligible=[]
    shifts=list(itertools.product([-.03,-.015,0,.015,.03],[-.015,0,.015],[-.03,-.015,0,.015,.03]))
    shifts.sort(key=lambda s:sum(x*x for x in s))
    for seed in seeds:
        for degrees in (90,-90):
            rotation=np.eye(4);rotation[:3,:3]=Rotation.from_euler('z',degrees,degrees=True).as_matrix()
            base=np.array(seed['mesh_T_gripper'])@rotation
            for shift in shifts:
                t=base.copy();t[:3,3]+=t[:3,:3]@shift
                assert np.allclose(t[:3,2],np.array(seed['mesh_T_gripper'])[:3,2])
                row=dict(parent_reference=seed['id'],local_z_roll_degrees=degrees,
                         translation_in_rolled_graspgen_frame_m=list(shift),mesh_T_gripper=t.tolist())
                reason=None
                vertices=trimesh.transform_points(hand.vertices,t)
                clearance=min(vertices[:,2].min(),vertices[:,2].min()-.08*t[2,2])-mesh.bounds[0,2]
                if clearance<.002: reason='table_clearance'
                if reason is None:
                    for depth in np.linspace(0,.08,17):
                        approach=t.copy();approach[:3,3]-=depth*t[:3,2];manager.set_transform('hand',approach)
                        if any(manager.in_collision_other(cm) for cm in obstacles):
                            reason='open_hand_or_approach_collision';break
                if reason is None:
                    hits=[c.query(t) for c in contacts];row['contacts']=dict(scan=hits[0],cooked=hits[1])
                    if any(c is None for h in hits for c in h):reason='missing_two_finger_contacts'
                    elif any(c['q_m']>=.03999 for h in hits for c in h):reason='contact_while_open'
                    elif any(abs(a['q_m']-b['q_m'])>.002 for a,b in zip(*hits)):reason='contact_surface_disagreement'
                    else:
                        if name=='mug':
                            handle=all(c['point_m'][0]>.021 for h in hits for c in h)
                        else:
                            d=np.array([.071,.069,0]);d/=np.linalg.norm(d)
                            split=np.array([-.015,.03,0])+d*.068
                            handle=all((np.array(c['point_m'])-split)@d>0 for h in hits for c in h)
                        row['both_contacts_on_handle']=bool(handle)
                        if not handle:reason='contacts_not_both_on_handle'
                row['result']=reason or 'geometry_pass';counts[row['result']]+=1
                if reason is None:eligible.append(len(records))
                records.append(row)
        print(name,seed['id'],dict(counts),flush=True)
    report=dict(status='geometry_screened',sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in
        (source/'bank.json',gallery,host(obj['mesh']),host(obj['cooked']),ROOT/'graspgen/try_handle_roll.py')},
        seed_count=len(seeds),generated=len(records),counts=dict(counts),records=records,
        method='Exact local Z +/-90 degrees; unchanged approach axis; local translations up to 30/15/30 mm; open cooked hand 8 cm approach sampled every 5 mm; scan+cooked independent contacts within 2 mm. No GraspGen score inherited.',
        scope='Geometry only; not physical success or expert trajectory validation.')
    save(out/'geometry-report.json',report)
    if not eligible:return
    poses=np.array([records[i]['mesh_T_gripper'] for i in eligible])
    # Neutral deterministic preference for smallest displacement; this is NOT a learned score.
    quality=-np.array([np.linalg.norm(records[i]['translation_in_rolled_graspgen_frame_m']) for i in eligible])
    # Opposite +/-90 rolls can be equivalent under finger-swap symmetry.
    for count in range(min(16,len(eligible)),0,-1):
        try:
            indices=farthest_selection(poses,quality,count)
            break
        except ValueError:
            continue
    refs=[]
    for j,k in enumerate(indices):
        r=records[eligible[k]];t=np.array(r['mesh_T_gripper']);h=t@HAND_OFFSET
        refs.append(dict(id=f'{name}-roll90-{j:03d}',position=h[:3,3].tolist(),
            quaternion=Rotation.from_matrix(h[:3,:3]).as_quat()[[3,0,1,2]].tolist(),
            **{k:v for k,v in r.items() if k!='result'}))
    obj['references']=refs;bank['status']='isolated_roll90_geometry_candidates_not_training_ready'
    save(out/'bank.json',bank);save(out/'references.json',refs)
    prepare_mask(source,out,bank,mesh,parts)

def prepare_mask(source,out,bank,mesh,parts):
    name=bank['objects'][0]['name'];refs=bank['objects'][0]['references']
    mask=read(source/'position-mask.json')
    obstacles=CollisionManager()
    for n,(record,m) in enumerate(pickle.load((ROOT/'assets/kitchen/scene_meshes.pkl').open('rb'))):
        obstacles.add_object(str(n),m.convex_hull if record['type']=='Mesh' else m)
    rz=np.eye(4);rz[:3,:3]=Rotation.from_euler('z',-90,degrees=True).as_matrix()
    # Continuous conservative furniture sweep across each existing 1 cm window + 2 mm margin.
    shifts=np.array(list(itertools.product([-.007,.007],[-.007,.007],[-.002,.002])))
    sweeps=[]
    for r in refs:
        t=rz@np.array(r['mesh_T_gripper'])@HAND_OFFSET;group=[]
        for part in parts:
            v=trimesh.transform_points(part.vertices,t);v=np.vstack([v,v-.08*t[:3,2]])
            group.append(trimesh.convex.convex_hull((v[:,None,:]+shifts).reshape(-1,3)))
        sweeps.append(group)
    cells=[];regions=[];cases=[];excluded=[]
    for region,oldcell in zip(mask['regions'],mask['cells']):
        x,y=np.mean(oldcell['bounds_map_xy'],axis=0);t=np.eye(4)
        t[:3,3]=[y-6.35,.3-x,mask['bowl_origin_z_m']]
        allowed=[i for i,g in enumerate(sweeps) if not any(obstacles.in_collision_single(m,transform=t) for m in g)]
        if not allowed:excluded.append(region['name']);continue
        # Existing footprint is unchanged and already checked; only new hand sweep is re-evaluated.
        regions.append(dict(region,cell_offset=len(cells)))
        cells.append(dict(region=region['name'],bounds_map_xy=oldcell['bounds_map_xy'],allowed_reference_indices=allowed))
        for i in allowed:
            cases.append(dict(id=f'{refs[i]["id"]}-{region["name"]}',object=name,reference=refs[i]['id'],region=region['name'],depth='window-center',
                object_state=[*t[:3,3],2**-.5,0,0,-2**-.5,0,0,0,0,0,0],robot_state=[0,.05,.0259,1,0,0,0,0,0,0,0,0,0]))
    assert cells, 'No kitchen placement survived'
    fingerprints={k:v for k,v in mask['sources_sha256'].items() if k.startswith('assets/') or k.startswith('rl/')}
    fingerprints[str((source/'position-mask.json').relative_to(ROOT))]=sha(source/'position-mask.json')
    fingerprints[str((out/'bank.json').relative_to(ROOT))]=sha(out/'bank.json')
    newmask=dict(schema_version=1,status='validated',object_name=name,bank_path=str((out/'bank.json').relative_to(ROOT)),
        reference_ids=[r['id'] for r in refs],bowl_map_yaw_degrees=0.,bowl_origin_z_m=mask['bowl_origin_z_m'],cell_size_m=.01,
        numeric_position_tolerance_m=1e-6,sources_sha256=fingerprints,regions=regions,cells=cells,empty_cells=[],
        allowed_cells_per_reference=[sum(i in c['allowed_reference_indices'] for c in cells) for i in range(len(refs))],
        excluded_empty_windows=excluded,scope='Static hand/furniture clearance only; no inherited physical acceptance.',
        method='New cooked hand continuous 8 cm sweep over original 1 cm windows with 2 mm margin; unchanged asset footprint.')
    save(out/'position-mask.json',newmask);save(out/'placement-cases.json',cases)
    cfg=yaml.safe_load((source/'task.yaml').read_text())
    cfg.pop('expert_preflight',None);cfg.pop('evaluation',None)
    cfg['garmi_task']['multi_object'].update(manifest=inside(out/'bank.json'),reference_ids={name:[r['id'] for r in refs]})
    kit=cfg['garmi_task']['kitchen'];kit['mask']=inside(out/'position-mask.json');kit['files_sha256']['mask']=sha(out/'position-mask.json')
    kit['bowl_regions']=[dict(name=r['name'],bounds_map_xy=r['bounds_map_xy'],probability=1/len(regions)) for r in regions]
    (out/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print('PREPARED',name,'refs',len(refs),'cases',len(cases),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--object',choices=['mug','pitcher'],required=True)
    prepare(p.parse_args().object)
