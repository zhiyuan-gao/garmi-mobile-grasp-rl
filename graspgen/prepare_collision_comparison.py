"""Prepare isolated diagnostics: same reference poses, placement cells and controller."""
from pathlib import Path
import hashlib,json,yaml
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dump=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
for y in ['025_mug','019_pitcher_base']:
    old=ROOT/'runs/ycb-rl-validation'/y
    out=ROOT/'runs/ycb-collision-v2'/y;out.mkdir(exist_ok=False)
    asset=ROOT/'assets/ycb'/y/'collision-v2'
    bank=json.loads((old/'bank.json').read_text())
    bank['status']='collision_candidate_diagnostics_only_expert_not_authorized'
    obj=bank['objects'][0]
    obj.update(usd='/workspace/'+str((asset/'rl.usda').relative_to(ROOT)),usd_sha256=sha(asset/'rl.usda'))
    dump(out/'bank.json',bank)
    mask=json.loads((old/'position-mask.json').read_text())
    del mask['sources_sha256'][str((old/'bank.json').relative_to(ROOT))]
    mask['bank_path']=str((out/'bank.json').relative_to(ROOT))
    mask['sources_sha256'][str((out/'bank.json').relative_to(ROOT))]=sha(out/'bank.json')
    dump(out/'position-mask.json',mask)
    cfg=yaml.safe_load((old/'task.yaml').read_text())
    cfg['garmi_task']['multi_object']['manifest']='/workspace/'+str((out/'bank.json').relative_to(ROOT))
    cfg['garmi_task']['kitchen']['mask']='/workspace/'+str((out/'position-mask.json').relative_to(ROOT))
    cfg['garmi_task']['kitchen']['files_sha256']['mask']=sha(out/'position-mask.json')
    (out/'task.yaml').write_text(yaml.safe_dump(cfg))
    dump(out/'comparison-scope.json',dict(original_directory=str(old.relative_to(ROOT)),
        changed='Object physical collision shapes; PhysX recomputes inertia from these shapes.',
        unchanged=['40 reference poses','placement cells','mass','friction','controller','random seed','source-mesh static endpoint precheck'],
        limitations=['Static endpoint precheck still uses source surface for direct comparability; it does not certify new proxy clearance. Physical forbidden contact checks remain active.',
            'Masks check hand/furniture clearance. Whole-object placement clearance audited separately.'],
        expert_trajectories_authorized=False,training_authorized=False))
