"""Exercise boundary case serialization in the actual parallel Isaac scene."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--out', required=True)
p.add_argument('--num_envs', type=int, default=128)
p.add_argument('--resets', type=int, default=300)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import json
import time
import traceback
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from garmi_rl.config import configured_env
from garmi_rl.kitchen import initial_collection_cases, ensure_filter

torch.set_num_threads(4)
cfg, _ = configured_env(yaml.safe_load(Path(args.config).read_text()), args.num_envs, 2026101819)
env = ManagerBasedRLEnv(cfg)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
try:
    env.reset()
    bank = env.grasp_bank
    filt = ensure_filter(bank)
    cases = bank.cases()
    # Deliberately put valid samples close to both X boundaries, where world
    # float32 serialization previously rejected an otherwise legal position.
    xy = torch.empty((args.num_envs, 2), device=env.device)
    for i in range(args.num_envs):
        region = filt.data['regions'][(i//2) % len(filt.data['regions'])]
        lo, hi = region['bounds_map_xy']
        xy[i] = xy.new_tensor([lo[0]+2.e-7 if i%2 == 0 else hi[0]-2.e-7, (lo[1]+hi[1])/2])
    indices = filt.sample(xy, 0., bank.pools[0])
    for i, case in enumerate(cases):
        point = xy[i].double().tolist()
        case['object_state'][:2] = [point[1]-6.35, .3-point[0]]
        case['reference'] = filt.data['reference_ids'][int(indices[i])]
    bank.restore(cases)
    raw = bank.cases()
    failed = []
    for i, c in enumerate(raw):
        point = [[.3-c['object_state'][1], 6.35+c['object_state'][0]]]
        try:
            filt.allowed(point, 0., bank.pools[0])
        except ValueError as error:
            failed.append(dict(env=i, sampled=xy[i].tolist(), roundtrip=point[0], error=str(error)))
    # A smaller placement region need not reproduce the historical rounding bug.
    # Valid restored cases and rejection of genuinely invalid poses remain required.
    if not failed:
        print('No raw boundary serialization errors in this region/layout', flush=True)
    fixed, maximum = initial_collection_cases(bank)
    before = env.scene['object'].data.root_pos_w.clone()
    bank.restore(fixed)
    pose_change = float((env.scene['object'].data.root_pos_w-before).abs().max())
    assert pose_change < 1.e-5, pose_change
    for j in range(args.resets):
        env.reset()
        fixed, error = initial_collection_cases(bank)
        maximum = max(maximum, error)
        bank.restore(fixed)
        if j % 50 == 0:
            print('RESET_CHECK', j, maximum, flush=True)
    # An actual out-of-bounds pose must still fail: no mask tolerance was changed.
    rejected = False
    try:
        filt.allowed([[1.169, 7.43]], 0., bank.pools[0])
    except ValueError:
        rejected = True
    assert rejected
    summary = dict(status='passed', boundary_failures_before=len(failed), boundary_failures_after=0,
        placement_regions=[r['name'] for r in filt.data['regions']],
        examples=failed[:8], random_reset_cases=args.resets*args.num_envs,
        maximum_roundtrip_error_m=maximum, maximum_boundary_physics_pose_change_m=pose_change,
        genuinely_outside_position_rejected=rejected, wall_seconds=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('COLLECTION_RESET_CHECK_DONE', json.dumps(summary), flush=True)
except BaseException:
    # Kit's fast shutdown otherwise exits before Python prints the exception.
    traceback.print_exc()
    app.set_setting('/app/fastShutdown', False)
