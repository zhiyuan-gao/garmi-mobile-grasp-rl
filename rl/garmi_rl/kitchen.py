"""Kitchen scene and validated position-dependent grasp sampling.

Opted into by garmi_task.kitchen; no process-wide monkey patches. Geometry and
grasp references are the frozen, previously validated preflight artifacts.
"""
import copy
import hashlib
import json
from pathlib import Path

import torch
from .reference_position_filter import ReferencePositionFilter

ROOT = Path(__file__).resolve().parents[2]


def configure(cfg, settings):
    import isaaclab.sim as sim
    cfg.kitchen = copy.deepcopy(settings)
    if cfg.multi_object["objects"] != ["bowl"] or cfg.multi_object.get("split") != "all":
        raise ValueError("Kitchen task requires the fixed bowl bank, split=all")
    for key in ("base_ranges", "object_xy", "object_yaw_degrees", "reset_corners"):
        if key in cfg.multi_object:
            raise ValueError(f"Kitchen sampling is configured by kitchen, not multi_object.{key}")
    if settings["bowl_map_yaw_degrees"] != 0 or settings["base_map_yaw_degrees"] != 90:
        raise ValueError("Validated kitchen setup requires bowl yaw 0 and initial base yaw 90")
    for key, expected in settings["files_sha256"].items():
        if hashlib.sha256(Path(settings[key]).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen kitchen input changed: {key}")
    cfg.scene.env_spacing = 10.
    cfg.scene.table.spawn = sim.UsdFileCfg(usd_path=settings["scene_usd"], activate_contact_sensors=True)
    cfg.scene.table.init_state.pos = (0., 0., 0.)
    cfg.viewer.eye = (-1.8, .5, 2.4)
    cfg.viewer.lookat = (.85, -.4, 1.)
    return dict(**copy.deepcopy(settings), reset_order="bowl position, then uniform allowed reference",
                references_fixed_per_episode=True, self_collision=True,
                furniture_collision=True, object_attachment=False)


def ensure_filter(bank):
    if not hasattr(bank, "position_filter"):
        bank.position_filter = ReferencePositionFilter(bank.env.cfg.kitchen["mask"],
            bank.settings["manifest"], ROOT, bank.env.device)
        bank.sampled_map_xy = torch.zeros(bank.env.num_envs, 2, device=bank.env.device)
    return bank.position_filter


def reset_filtered(bank, ids):
    env = bank.env
    settings = env.cfg.kitchen
    position_filter = ensure_filter(bank)
    random = torch.rand(len(ids), 6, device=env.device)
    root = env.scene["robot"].data.default_root_state[ids].clone()
    base_lo, base_hi = torch.tensor(settings["base_bounds_map_xy"], device=env.device)
    base = base_lo + random[:, :2] * (base_hi-base_lo)
    root[:, 0], root[:, 1] = base[:, 1]-6.35, .3-base[:, 0]
    root[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=env.device)
    root[:, 7:] = 0.
    regions = settings["bowl_regions"]
    right = random[:, 2] >= regions[0]["probability"]
    bounds = torch.tensor([r["bounds_map_xy"] for r in regions], device=env.device)[right.long()]
    map_xy = bounds[:, 0] + random[:, 3:5] * (bounds[:, 1]-bounds[:, 0])
    chosen = position_filter.sample(map_xy, 0., bank.pools[0], random[:, 5])
    bank.sampled_map_xy[ids] = map_xy
    bank.reference_ids[ids] = chosen
    env.grasp_pos_o[ids] = bank.tables[0][chosen, :3]
    env.grasp_quat_o[ids] = bank.tables[0][chosen, 3:]
    obj = env.scene["object"].data.default_root_state[ids].clone()
    obj[:, 0], obj[:, 1] = map_xy[:, 1]-6.35, .3-map_xy[:, 0]
    obj[:, 2] = position_filter.data["bowl_origin_z_m"]
    obj[:, 3:7] = torch.tensor([2.**-.5, 0., 0., -2.**-.5], device=env.device)
    obj[:, 7:] = 0.
    root[:, :3] += env.scene.env_origins[ids]
    obj[:, :3] += env.scene.env_origins[ids]
    return root, obj


def validate_cases(bank, cases):
    """Reject incompatible saved cases before restoring them to physics."""
    position_filter = ensure_filter(bank)
    settings = bank.env.cfg.kitchen
    obj = torch.tensor([c["object_state"] for c in cases], dtype=torch.float64, device=bank.env.device)
    base = torch.tensor([c["robot_state"] for c in cases], dtype=torch.float64, device=bank.env.device)
    xy = torch.stack((.3-obj[:, 1], 6.35+obj[:, 0]), dim=1)
    q = obj[:, 3:7]
    yaw = torch.rad2deg(torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                                  1-2*(q[:, 2]**2+q[:, 3]**2))) + 90.
    allowed = position_filter.allowed(xy, yaw, bank.pools[0])
    indices = torch.tensor([position_filter.data["reference_ids"].index(c["reference"])
                            for c in cases], device=bank.env.device)
    if not allowed[torch.arange(len(cases), device=bank.env.device), indices].all():
        raise ValueError("Saved case uses a masked reference")
    if (obj[:, 2]-position_filter.data["bowl_origin_z_m"]).abs().max() > 1.e-4:
        raise ValueError("Saved bowl height differs from the validated mask")
    xy_base = torch.stack((.3-base[:, 1], 6.35+base[:, 0]), dim=1)
    lo, hi = torch.tensor(settings["base_bounds_map_xy"], device=bank.env.device)
    if ((xy_base < lo-1.e-4) | (xy_base > hi+1.e-4)).any():
        raise ValueError("Saved base pose outside the initial region")
    if not torch.allclose(base[:, 3:7].abs(), base.new_tensor([1., 0., 0., 0.]).expand(len(cases), -1), atol=1.e-5):
        raise ValueError("Saved base must initially face the counter")
    bank.sampled_map_xy[:len(cases)] = xy.float()


def initial_collection_cases(bank):
    """Export freshly reset bowl XY from the sampled pose, before world rounding.

    Float32 world coordinates include large parallel-environment offsets. Removing
    those offsets can move a boundary point outside the mask's 1 um tolerance.
    This helper is only valid immediately after reset, never after an action.
    The caller restores these cases before collecting observations/actions.
    """
    cases = bank.cases()
    xy = bank.sampled_map_xy.double().cpu().tolist()
    maximum_error = 0.
    for case, point in zip(cases, xy):
        expected = [point[1] - 6.35, .3 - point[0]]
        error = max(abs(a-b) for a,b in zip(case['object_state'][:2], expected))
        maximum_error = max(maximum_error, error)
        # This is a serialization fix, not a way to repair a moved object.
        if error > 5.e-5:
            raise ValueError('Initial sampled bowl pose differs from physics by more than 50 um')
        case['object_state'][:2] = expected
    return cases, maximum_error
