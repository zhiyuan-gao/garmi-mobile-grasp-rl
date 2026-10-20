"""Load retained-grasp configs and exercise reset/restore only; never plan or step actions."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(__doc__)
p.add_argument('--config',required=True);p.add_argument('--out',required=True)
p.add_argument('--num_envs',type=int,default=80)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app

import collections
import hashlib
import json
import traceback
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from garmi_rl.config import configured_env
from garmi_rl.kitchen import initial_collection_cases,ensure_filter

env=None
try:
    torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    profile=yaml.safe_load(Path(args.config).read_text())
    cfg,_=configured_env(profile,args.num_envs,2026101902)
    env=ManagerBasedRLEnv(cfg);env.forbidden_sensors=cfg.forbidden_sensors
    seen=collections.Counter();maximum=0.
    for _ in range(10):
        env.reset();bank=env.grasp_bank
        cases,error=initial_collection_cases(bank);maximum=max(maximum,error)
        bank.restore(cases);seen.update(c['reference'] for c in cases)
    filt=ensure_filter(bank)
    audit=json.loads(Path(profile['expert_preflight']['audit_cases']).read_text())
    evaluation=json.loads(Path(profile['evaluation']['cases']).read_text())
    for case_group in [audit,evaluation]:
        for start in range(0,len(case_group),args.num_envs):
            chunk=case_group[start:start+args.num_envs]
            bank.restore(chunk+[chunk[-1]]*(args.num_envs-len(chunk)))
    assert set(seen)==set(filt.data['reference_ids'])
    outside_rejected=False
    try:filt.allowed([[-2.,5.]],0.,bank.pools[0])
    except ValueError:outside_rejected=True
    assert outside_rejected
    report=dict(status='passed',reset_cases=10*args.num_envs,
        sampled_reference_counts=dict(seen),maximum_roundtrip_error_m=maximum,
        audit_initial_states_checked=len(audit),evaluation_initial_states_checked=len(evaluation),
        outside_regions_rejected=True,configuration_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        expert_trajectories_generated=0,optimizer_updates=0,action_steps=0,
        scope='Actual asset load, masses/hashes, subset mask, restricted reset, serialization and restore. No approach or grasp execution.')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RETAINED_INPUTS_CHECKED',json.dumps(report),flush=True)
except BaseException:
    traceback.print_exc();app.set_setting('/app/fastShutdown',False);raise
finally:
    if env is not None:env.close()
    app.close()
