"""GAP-RL-inspired pose shaping for the fixed-finish reward ablation.

Reference: THU-VCLab/GAP-RL, commit 7fc984f363621efcccffe375be83dd1c89b6c360,
gap_rl/envs/pick_single.py:1089-1205. This is not a full GAP-RL reproduction.
"""
import torch


def pose_terms(distance, angle, distance_clip_m=.5, distance_scale_per_m=5.):
    """Return position/orientation scores and the quaternion rotation distance.

    For the shortest quaternion angle, 1-cos(angle/2) == 1-|dot(q, q_goal)|.
    Use the assigned goal, without finger-exchange symmetry, to agree with the
    existing handover criterion. Keep the source's 0.5 upper clipping; remove
    its 0.01 lower clipping so alignment below 0.15 rad remains rewarded.
    Divide the source's final approach reward by 3 to retain a unit maximum.
    """
    translation = distance.clamp(0., distance_clip_m)
    rotation = (1. - torch.cos(angle * .5)).clamp(0., .5)
    position_score = 1. - torch.tanh(distance_scale_per_m * translation)
    rotation_score = 1. - torch.tanh(5. * rotation)
    return position_score, rotation_score, rotation


def policy_approach(env, distance_clip_m=.5, distance_scale_per_m=5.):
    from .mdp import diagnostics
    d = diagnostics(env)
    position, rotation, _ = pose_terms(d["distance"], d["angle"], distance_clip_m, distance_scale_per_m)
    return .5 * (position + rotation) * (env.fixed_finish.phase == 0)


def description():
    return dict(
        name="gap_pose_v1",
        source="https://github.com/THU-VCLab/GAP-RL/blob/7fc984f363621efcccffe375be83dd1c89b6c360/gap_rl/envs/pick_single.py#L1089-L1205",
        formula="0.5*(1-tanh(5*clip(d,0,0.5))) + 0.5*(1-tanh(5*clip(1-cos(theta/2),0,0.5)))",
        position_units="meters", angle_units="radians", raw_maximum=1.,
        normalization="Source approach component after its final x0.5 scaling, divided by 3",
        adaptations=["Assigned episode reference, not nearest grasp in a set",
                     "Exact assigned orientation, without finger-exchange symmetry",
                     "Remove 0.01 lower clipping to resolve the existing 0.15 rad handover threshold"],
        unchanged=["PPO", "observations/actions", "reset distribution", "physics/contacts",
                   "fixed finish", "success/failure criteria", "success bonus 115", "action-rate penalty"])
