"""Opt-in reward used only by the isolated pitcher PPO ablation."""
import math
import torch
from .gap_rewards import pose_terms


def precision_score(distance, angle, coarse_weight=.2, position_sigma_m=.02,
                    angle_sigma_rad=.10, distance_clip_m=2., distance_scale_per_m=2.):
    values = (coarse_weight, position_sigma_m, angle_sigma_rad,
              distance_clip_m, distance_scale_per_m)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Reward parameters must be finite")
    if not 0 <= coarse_weight <= 1 or min(values[1:]) <= 0:
        raise ValueError("Invalid reward weights or scales")
    position, rotation, _ = pose_terms(distance, angle, distance_clip_m, distance_scale_per_m)
    fine = torch.exp(-.5 * ((distance / position_sigma_m).square()
                          + (angle / angle_sigma_rad).square()))
    return coarse_weight * .5 * (position + rotation) + (1 - coarse_weight) * fine


def policy_approach(env, coarse_weight=.2, position_sigma_m=.02, angle_sigma_rad=.10,
                    distance_clip_m=2., distance_scale_per_m=2.):
    from .mdp import diagnostics
    d = diagnostics(env)
    return precision_score(d["distance"], d["angle"], coarse_weight, position_sigma_m,
                           angle_sigma_rad, distance_clip_m,
                           distance_scale_per_m) * (env.fixed_finish.phase == 0)


def description(parameters):
    return dict(name="pitcher_precision_v1", parameters=parameters, raw_maximum=1.,
                formula="0.2*gap_pose_v1 + 0.8*exp(-0.5*((d/0.02)^2+(theta/0.10)^2))",
                position_units="meters", angle_units="radians",
                shaping_only_during_approach=True)
