"""Screen fresh mug candidates or prepare an isolated GARMI kitchen diagnostic."""
import argparse,json,pickle
from pathlib import Path
import numpy as np,trimesh,yaml
from select_references import ROOT,HAND_OFFSET,screen,sha,verify_bowl_conversion
from prepare_rl_validation import hand_parts,build_mask,crop_to_supported_footprint,save,MASS_SOURCE

OUT=ROOT/'runs/mug-surface-v3/025_mug'
ASSET=ROOT/'assets/ycb/025_mug/surface-v3'
inside=lambda p:'/workspace/'+str(p.relative_to(ROOT))


def screen_candidates():
    pool=np.load(OUT/'candidate_pool.npz')
    poses,scores=pool['T_object_graspgen'],pool['scores']
    hand=trimesh.util.concatenate(hand_parts());hand.apply_transform(HAND_OFFSET)
    surface=trimesh.load(ASSET/'repaired.obj',force='mesh',process=False)
    cooked=trimesh.util.concatenate([p for _,p in pickle.load((ASSET/'cooked.pkl').open('rb'))])
    fingerprints=dict(pool_sha256=sha(OUT/'candidate_pool.npz'),mesh_sha256=sha(ASSET/'repaired.obj'),
        cooked_sha256=sha(ASSET/'cooked.pkl'),robot_cooked_sha256=sha(ROOT/'assets/kitchen/cooked-model.pkl'))
    valid=np.ones(len(poses),dtype=bool);reports={};clearances=[]
    for name,mesh in [('sampling-surface',surface),('physical-collision',cooked)]:
        target=OUT/name;target.mkdir(exist_ok=False)
        v,c,counts=screen(mesh,hand,poses,scores,target,fingerprints)
        valid &= v;clearances.append(c);reports[name]=counts
    np.savez_compressed(OUT/'geometry-screen.npz',valid=valid,plane_clearance_m=np.minimum(*clearances))
    save(OUT/'geometry-screen.json',dict(sources=fingerprints,counts=reports,eligible_both=int(valid.sum()),
        method='Full actual cooked wrist/palm/fingers against repaired surface AND exported physical convexes; original score, table and approach thresholds.'))
    print('SCREEN_COMPLETE',int(valid.sum()),reports,flush=True)


def prepare_task():
    verify_bowl_conversion()
    refs=json.loads((OUT/'selected-40.json').read_text())
    assert len(refs)==40 and len({r['id'] for r in refs})==40
    mesh=trimesh.load(ASSET/'repaired.obj',force='mesh',process=False)
    bank_path=OUT/'bank.json'
    assert not bank_path.exists()
    save(bank_path,dict(status='repaired_surface_diagnostic_only_expert_pending',objects=[dict(name='mug',
        mesh=inside(ASSET/'repaired.obj'),mesh_sha256=sha(ASSET/'repaired.obj'),
        usd=inside(ASSET/'rl.usda'),usd_sha256=sha(ASSET/'rl.usda'),
        mass=.118,mass_source=MASS_SOURCE,friction_assumption=1.,bounds=mesh.bounds.tolist(),references=refs)]))
    mask=crop_to_supported_footprint(build_mask('mug',mesh,refs,bank_path),mesh)
    save(OUT/'position-mask.json',mask)
    assert mask['status']=='validated',f"Empty cells: {mask['empty_cells']}"
    cfg=yaml.safe_load((ROOT/'configs/task.yaml').read_text());task=cfg['garmi_task']
    task['multi_object']=dict(manifest=inside(bank_path),objects=['mug'],split='all',reference_ids={'mug':[r['id'] for r in refs]})
    task['kitchen']['mask']=inside(OUT/'position-mask.json')
    for region,valid in zip(task['kitchen']['bowl_regions'],mask['regions']):region['bounds_map_xy']=valid['bounds_map_xy']
    task['kitchen']['files_sha256']=dict(mask=sha(OUT/'position-mask.json'),scene_usd=sha(ROOT/'assets/kitchen/kitchen.usda'))
    (OUT/'task.yaml').write_text(yaml.safe_dump(cfg))
    print('PREPARED',mask['allowed_cells_per_reference'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('step',choices=['screen','prepare']);args=p.parse_args()
    screen_candidates() if args.step=='screen' else prepare_task()
