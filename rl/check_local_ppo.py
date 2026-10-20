"""Exercise full and partial training-pool resets in the actual Isaac scene."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
p.add_argument('--num_envs',type=int,default=256);AppLauncher.add_app_launcher_args(p)
a=p.parse_args();app=AppLauncher(a).app
import json,traceback
import torch,yaml
from isaaclab.envs import ManagerBasedRLEnv
from garmi_rl.config import configured_env
cfg,_=configured_env(yaml.safe_load(Path(a.config).read_text()),a.num_envs,2026102001)
env=ManagerBasedRLEnv(cfg);env.forbidden_sensors=cfg.forbidden_sensors
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
try:
 assert cfg.require_strict_success
 maximum=0.;refs=set();total=0
 for iteration in range(10):
  if iteration%2==0:env.reset();ids=torch.arange(env.num_envs,device=env.device)
  else:
   ids=torch.arange(0,env.num_envs,2,device=env.device);env._reset_idx(ids)
  bank=env.grasp_bank;chosen=bank.local_reset_indices[ids];pool=bank.local_reset_pool
  for field,asset in [('robot','robot'),('object','object')]:
   actual=env.scene[asset].data.root_state_w[ids].clone();actual[:,:3]-=env.scene.env_origins[ids]
   error=float((actual-pool[field][chosen]).abs().max());maximum=max(maximum,error);assert error<3.e-5,error
  assert torch.equal(bank.reference_ids[ids],pool['references'][chosen])
  refs.update(bank.reference_ids[ids].cpu().tolist());total+=len(ids)
  obs=env.observation_manager.compute()['policy'];assert torch.isfinite(obs).all()
 assert len(refs)==40
 # Exercise actual physics/termination calls as well as pose restoration.
 for _ in range(5):
  _,reward,_,_,_=env.step(torch.zeros((env.num_envs,10),device=env.device));assert torch.isfinite(reward).all()
 report=dict(status='passed',reset_cases=total,references=len(refs),maximum_state_roundtrip_error=maximum,
             partial_resets_verified=True,strict_reward_enabled=True,training_reset_pool_sha256=bank.settings['training_reset_pool']['sha256'])
 (out/'summary.json').write_text(json.dumps(report,indent=2));print('LOCAL_PPO_RESET_PASSED',json.dumps(report),flush=True)
except BaseException:
 traceback.print_exc();app.set_setting('/app/fastShutdown',False);raise
finally:env.close();app.close()
