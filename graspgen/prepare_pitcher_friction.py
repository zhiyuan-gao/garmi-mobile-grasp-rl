"""Version pitcher-only friction changes and rebind matching validation inputs."""
import argparse
import json
import shutil
from pathlib import Path

import yaml
from pxr import Usd, UsdPhysics
from select_references import ROOT, sha
from prepare_rl_validation import save


def inside(path):
    return '/workspace/'+str(path.relative_to(ROOT))


def main(source, target, config):
    parent=ROOT/'assets/ycb/019_pitcher_base/contact-v3-coarse'
    asset=ROOT/'assets/ycb/019_pitcher_base/contact-v5-friction'
    original=(parent/'rl.usda').read_text()
    updated=original
    for field in ['staticFriction','dynamicFriction']:
        old=f'float physics:{field} = 1\n'
        assert updated.count(old)==1
        updated=updated.replace(old,f'float physics:{field} = 2\n')
    if asset.exists():
        assert (asset/'rl.usda').read_text()==updated
    else:
        asset.mkdir()
        (asset/'rl.usda').write_text(updated)
        shutil.copy2(parent/'cooked.pkl',asset/'cooked.pkl')
        meta=json.loads((parent/'cooked.json').read_text())
        meta.update(usd_sha256=sha(asset/'rl.usda'),geometry_reused_from=str(parent.relative_to(ROOT)),
                    reason='Only two material scalar values changed; geometry and physics shape settings are byte-identical.')
        save(asset/'cooked.json',meta)
        save(asset/'material-change.json',dict(parent_usd_sha256=sha(parent/'rl.usda'),
            usd_sha256=sha(asset/'rl.usda'),static_friction=2.,dynamic_friction=2.,
            robot_static_dynamic_friction=.5,combine_mode='average',
            expected_robot_contact_friction=1.25,expected_worktop_contact_friction=1.5,
            previous_robot_contact_friction=.75,previous_worktop_contact_friction=1.,
            geometry_and_mass_unchanged=True,real_material_calibrated=False,
            authorization='User requested increased robot-pitcher friction relative to bowl.'))
    stage=Usd.Stage.Open(str(asset/'rl.usda'))
    material=UsdPhysics.MaterialAPI(stage.GetPrimAtPath('/Object/physicsMaterial'))
    assert material.GetStaticFrictionAttr().Get()==material.GetDynamicFrictionAttr().Get()==2.
    assert sha(asset/'cooked.pkl')==sha(parent/'cooked.pkl')
    target.mkdir(parents=True,exist_ok=False)
    assert not config.exists()
    bank=json.loads((source/'bank.json').read_text())
    assert len(bank['objects'])==1 and bank['objects'][0]['name']=='pitcher'
    obj=bank['objects'][0]
    assert obj['usd_sha256']==sha(parent/'rl.usda')
    obj.update(usd=inside(asset/'rl.usda'),usd_sha256=sha(asset/'rl.usda'),friction_assumption=2.)
    bank['status']='new_pitcher_friction_profile_references_require_physical_revalidation'
    save(target/'bank.json',bank)
    mask=json.loads((source/'position-mask.json').read_text())
    for path,digest in mask['sources_sha256'].items():assert sha(ROOT/path)==digest
    del mask['sources_sha256'][mask['bank_path']]
    mask['bank_path']=str((target/'bank.json').relative_to(ROOT))
    mask['sources_sha256'][mask['bank_path']]=sha(target/'bank.json')
    save(target/'position-mask.json',mask)
    cfg=yaml.safe_load((source/'task.yaml').read_text());task=cfg['garmi_task']
    task['multi_object']['manifest']=inside(target/'bank.json')
    task['kitchen']['mask']=inside(target/'position-mask.json')
    task['kitchen']['files_sha256']['mask']=sha(target/'position-mask.json')
    cfg['agent']['experiment']['experiment_name']='pitcher_friction_v5'
    config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(yaml.safe_dump(cfg))
    if config!=target/'task.yaml':shutil.copy2(config,target/'task.yaml')
    save(target/'preparation.json',dict(status='material_changed_physical_revalidation_pending',
        source=str(source.relative_to(ROOT)),source_bank_sha256=sha(source/'bank.json'),
        configuration=str(config.relative_to(ROOT)),configuration_sha256=sha(config),
        candidate_reference_count=len(obj['references']),
        validation_status='Old-profile successful labels are not transferred to this profile.',
        asset=str(asset.relative_to(ROOT)),static_friction=2.,dynamic_friction=2.,
        expected_robot_contact_friction=1.25,expected_worktop_contact_friction=1.5,
        collision_geometry_unchanged=True,placement_cells_unchanged=True,
        expert_trajectories_generated=0,optimizer_updates=0))
    print('PITCHER_FRICTION_PROFILE',config,len(obj['references']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    main(args.source.resolve(),args.out.resolve(),args.config.resolve())
