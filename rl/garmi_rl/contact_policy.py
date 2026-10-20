"""Target-specific contact permissions; furniture and self collision stay forbidden."""
import torch

POLICIES = ('fingers_only', 'fingers_and_wrist')


def permitted_forces(forces, wrist_matrix, object_filter_index, wrist_sensor_index, policy):
    """Apply permission to exactly the wrist/target pair, preserving other pairs.

    forces: [environment, sensor]; wrist_matrix: [environment, history, body,
    target, xyz]. Raw force tensors are never modified.
    """
    if policy not in POLICIES:
        raise ValueError(f'Unknown contact policy: {policy}')
    by_target = wrist_matrix.norm(dim=-1).amax(dim=(1, 2))
    target_force = by_target[:, object_filter_index]
    if policy == 'fingers_only':
        return forces, target_force
    others = [i for i in range(by_target.shape[1]) if i != object_filter_index]
    if not others:
        raise ValueError('Wrist sensor must retain non-target collision filters')
    permitted = forces.clone()
    permitted[:, wrist_sensor_index] = by_target[:, others].amax(dim=1)
    return permitted, target_force


def collision_forces(env):
    forces = torch.stack([env.scene[k].data.force_matrix_w_history.norm(dim=-1).flatten(1).amax(dim=1)
                          for k in env.forbidden_sensors], dim=1)
    policy = getattr(env.cfg, 'contact_policy', 'fingers_only')
    if policy == 'fingers_only':
        return forces, torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, '_wrist_target_filter'):
        keys = [k for k in env.forbidden_sensors
                if env.scene[k].cfg.prim_path.endswith('/right_fr3_link7')]
        if len(keys) != 1:
            raise ValueError('Expected exactly one wrist/palm collision sensor')
        key = keys[0]
        paths = env.scene[key].cfg.filter_prim_paths_expr
        indices = [i for i, p in enumerate(paths) if p.endswith('/Object')]
        if len(indices) != 1:
            raise ValueError('Expected exactly one target-object collision filter')
        env._wrist_target_filter = key, indices[0], len(paths)
    key, target, count = env._wrist_target_filter
    matrix = env.scene[key].data.force_matrix_w_history
    if matrix.shape[-2] != count:
        raise ValueError('Wrist contact tensor/filter order mismatch')
    return permitted_forces(forces, matrix, target, env.forbidden_sensors.index(key), policy)
