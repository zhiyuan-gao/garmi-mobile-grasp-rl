"""Opt-in PPO reset and strict alignment contracts for the validated local tasks."""
import hashlib
import json
from pathlib import Path


def strict_pose_gate(trigger, handover_distance, handover_angle, close_distance, close_angle):
    return ((trigger == 1) & (handover_distance <= .005) & (handover_angle <= .05)
            & (close_distance <= .005) & (close_angle <= .05))


def reset_from_training_pool(bank, ids):
    import torch
    from .kitchen import ensure_filter, validate_cases
    env = bank.env
    spec = bank.settings['training_reset_pool']
    if not hasattr(bank, 'local_reset_pool'):
        path = Path(spec['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
        doc = json.loads(path.read_text())
        assert doc['split'] == 'train' and doc['object'] == bank.objects[0]['name']
        assert len(bank.objects) == 1
        cases = doc['cases']
        ensure_filter(bank)
        # validate_cases also fills the per-environment map cache.
        for start in range(0, len(cases), env.num_envs):
            validate_cases(bank, cases[start:start+env.num_envs])
        mapping = {r['id']: i for i, r in enumerate(bank.references[0])}
        bank.local_reset_pool = dict(
            robot=torch.tensor([c['robot_state'] for c in cases], device=env.device),
            object=torch.tensor([c['object_state'] for c in cases], device=env.device),
            references=torch.tensor([mapping[c['reference']] for c in cases], device=env.device))
        bank.local_reset_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    pool = bank.local_reset_pool
    chosen = torch.randint(len(pool['references']), (len(ids),), device=env.device)
    bank.local_reset_indices[ids] = chosen
    refs = pool['references'][chosen]
    bank.reference_ids[ids] = refs
    env.grasp_pos_o[ids] = bank.tables[0][refs, :3]
    env.grasp_quat_o[ids] = bank.tables[0][refs, 3:]
    root, obj = pool['robot'][chosen].clone(), pool['object'][chosen].clone()
    bank.sampled_map_xy[ids] = torch.stack((.3-obj[:, 1], 6.35+obj[:, 0]), dim=1)
    root[:, :3] += env.scene.env_origins[ids]
    obj[:, :3] += env.scene.env_origins[ids]
    return root, obj
