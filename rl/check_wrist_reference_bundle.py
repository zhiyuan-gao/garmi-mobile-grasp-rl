"""Check all published references and mixed finger openings in the real Isaac scene."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(__doc__)
p.add_argument('--config',required=True);p.add_argument('--out',required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app

import hashlib,json,traceback
import torch,yaml
from isaaclab.envs import ManagerBasedRLEnv
from garmi_rl.config import configured_env
from garmi_rl import mdp
from garmi_rl.kitchen import initial_collection_cases

torch.set_num_threads(4)
config_path=Path(args.config);profile=yaml.safe_load(config_path.read_text())
cfg,_=configured_env(profile,40,2026101966);env=ManagerBasedRLEnv(cfg)
out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
try:
    env.reset();bank=env.grasp_bank
    assert cfg.contact_policy=='fingers_and_wrist'
    refs=bank.references[0];assert len(refs)==40
    cases=json.loads((config_path.parent/'audit-cases.json').read_text())
    assert len(cases)==40 and len({c['reference'] for c in cases})==40
    maximum=0.
    for _ in range(10):
        env.reset();serialized,error=initial_collection_cases(bank);bank.restore(serialized)
        maximum=max(maximum,error)
    bank.restore(cases)
    restored=bank.cases();obs=mdp.observation(env)
    table={r['id']:r for r in refs};widths=[]
    for i,c in enumerate(cases):
        assert restored[i]['reference']==c['reference']
        r=table[c['reference']];width=r.get('opening_m',cfg.fixed_finish.get('gripper_open_width_m',.08))
        assert abs(float(env.grasp_open_width[i])-width)<1e-7
        assert torch.max(torch.abs(env.scene['robot'].data.joint_pos[i,env.finger_joint_ids]-width/2))<1e-6
        assert torch.max(torch.abs(obs[i,7:9]+.04-width/2))<1e-6
        assert torch.max(torch.abs(env.grasp_pos_o[i]-torch.tensor(r['position'],device=env.device)))<1e-7
        assert torch.max(torch.abs(env.grasp_quat_o[i]-torch.tensor(r['quaternion'],device=env.device)))<1e-7
        widths.append(dict(reference=c['reference'],opening_m=width))
    summary=dict(status='passed',reference_count=40,random_reset_cases=400,maximum_roundtrip_error_m=maximum,
        per_reference_openings_and_observations_verified=True,contact_policy=cfg.contact_policy,
        configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),references=widths,
        scope='Runtime paths, reference indexing, reset/restore, and finger/observation consistency; physical successes are in validation.json')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('WRIST_BUNDLE_RUNTIME_CHECK_PASSED',flush=True)
except BaseException:
    traceback.print_exc();app.set_setting('/app/fastShutdown',False);raise
finally:
    env.close();app.close()
