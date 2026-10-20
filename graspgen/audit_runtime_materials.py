"""Read loaded physical materials without actions, expert collection or training."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(__doc__)
p.add_argument('--config', required=True)
p.add_argument('--out', required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import hashlib
import json
import torch
import yaml
from pxr import Usd, UsdPhysics, UsdShade
from isaaclab.envs import ManagerBasedRLEnv
from garmi_rl.config import configured_env

cfg, _ = configured_env(yaml.safe_load(Path(args.config).read_text()), 1, 2026101901)
env = ManagerBasedRLEnv(cfg)
try:
    env.reset()
    data = dict(config=args.config,
                config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                scene_default=dict(static=cfg.sim.physics_material.static_friction,
                                   dynamic=cfg.sim.physics_material.dynamic_friction,
                                   combine_mode=cfg.sim.physics_material.friction_combine_mode),
                tensor_column_order=['static_friction', 'dynamic_friction', 'restitution'],
                runtime_materials={}, finger_material_bindings=[], usd_materials=[],
                actions_executed=0, expert_trajectories_generated=0, optimizer_updates=0)
    for name in ['robot', 'object']:
        values = env.scene[name].root_physx_view.get_material_properties().detach().cpu()
        data['runtime_materials'][name] = dict(shape=list(values.shape),
            unique_values=torch.unique(values.reshape(-1, 3), dim=0).tolist())
    for prim in Usd.PrimRange.Stage(env.sim.stage, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            data['usd_materials'].append(dict(path=str(prim.GetPath()), attributes={
                a.GetName():a.Get() for a in prim.GetAttributes()
                if any(word in a.GetName().lower() for word in ['friction', 'restitution'])}))
        if prim.HasAPI(UsdPhysics.CollisionAPI) and 'finger' in str(prim.GetPath()):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            data['finger_material_bindings'].append(dict(path=str(prim.GetPath()),
                material=str(material.GetPath()) if material else None))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2)+'\n')
    print('RUNTIME_MATERIAL_AUDIT', json.dumps(data['runtime_materials']), flush=True)
finally:
    env.close()
    app.close()
