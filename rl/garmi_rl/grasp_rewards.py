"""Two-finger reward adaptation, usable on saved traces without starting Kit.

Formation follows MetaWorld's conditional closing / ManiSkill's grasp feedback
principle. The alignment gates and weights are GARMI trial parameters, not
FastGrasp constants. No action or physical attachment is imposed here.
"""
import torch

ALIGN_DISTANCE = 0.015
ALIGN_ANGLE = 0.15
OPEN_WIDTH = 0.08
WEIGHTS = dict(approach=1., close=1., grasp=2., lift=15., hold=5., action_rate=-1.e-4)


def formation_terms(distance, angle, width, contact, collision):
    aligned = (distance < ALIGN_DISTANCE) & (angle < ALIGN_ANGLE)
    close = aligned * ~contact * ~collision * (1. - width / OPEN_WIDTH).clamp(0., 1.)
    grasp = (contact & ~collision).float()
    return close, grasp


def positive_step_bound(step_dt):
    # Close and grasp are mutually exclusive. Negative action costs can only
    # lower the total. This is a conservative upper bound, not a typical reward.
    return step_dt * (WEIGHTS["approach"] + max(WEIGHTS["close"], WEIGHTS["grasp"])
                      + WEIGHTS["lift"] + WEIGHTS["hold"])


def completion_bonus(step_dt, discount):
    if not 0. <= discount < 1.:
        raise ValueError("Completion bound requires a discount in [0, 1)")
    return positive_step_bound(step_dt) / (1. - discount)


def describe(step_dt, discount, bonus_mode):
    if bonus_mode not in ("none", "discounted_upper_bound"):
        raise ValueError(f"Unknown completion reward: {bonus_mode}")
    bonus = completion_bonus(step_dt, discount) if bonus_mode == "discounted_upper_bound" else 0.
    return dict(variant="grasp_v2", alignment_distance=ALIGN_DISTANCE,
                alignment_angle=ALIGN_ANGLE, open_width=OPEN_WIDTH, weights=WEIGHTS,
                step_dt=step_dt, discount=discount, success_bonus_mode=bonus_mode,
                success_bonus=bonus, positive_step_bound=positive_step_bound(step_dt),
                contact_definition="both finger-object forces > 0.1 N; no forbidden collision")
