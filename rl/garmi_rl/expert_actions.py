"""Convert a geometric expert path to the existing 20 Hz action interface.

A DAgger query has no persistent trajectory clock. Restarting a rest-to-rest
cubic at every query makes its first position increment O(dt**2 / T**2), and
teaches the position servo to stop. Use a bounded displacement toward the next
waypoint instead, O(dt / T), and continue the previous joint target so drive
tracking errors do not accumulate. T is the planner's conservative duration,
not a new lookahead or a speed multiplier.

This is a geometric path follower, not a dynamics-aware planner. The physical
base acceleration limiter and joint drives remain in the environment. Recovery
validation must preserve measured velocities rather than reset them to zero.
"""
import numpy as np
from scipy.spatial.transform import Rotation

PARK = np.array([0., -.785398, 0., -2.356194, 0., 1.570796, .785398])
COMMAND_VERSION = "target_continuation_v4"
ARM_TARGET_LOWER = np.array([-2.75, -1.74, -2.75, -3., -2.73, .55, -2.89])
ARM_TARGET_UPPER = np.array([2.75, 1.74, 2.75, -.19, 2.73, 4.51, 2.89])


def first_action(query, plan):
    """One command toward the next waypoint, using the observed previous target."""
    a, b = np.asarray(plan['waypoints'][:2], dtype=np.float64)
    root = np.asarray(query['snapshot']['robot_state'], dtype=np.float64)
    rotation = Rotation.from_quat(root[[4, 5, 6, 3]])
    yaw = rotation.as_euler('xyz')[2]
    measured = np.r_[root[:2], yaw, query['snapshot']['arm_position']]
    difference = a - measured
    difference[2] = np.arctan2(np.sin(difference[2]), np.cos(difference[2]))
    if a.shape != (10,) or not np.isfinite([a, b]).all():
        raise ValueError('Expert edge must contain finite 10-dimensional states')
    if np.max(np.abs(difference)) > 2.e-5:
        raise ValueError('Plan does not start at the learner state')
    duration = float(plan['durations'][0])
    dt = float(query['step_dt'])
    if not np.isfinite([duration, dt]).all() or min(duration, dt) <= 0:
        raise ValueError('Expert duration and control period must be positive')
    delta = b - a
    delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
    # Continue the commanded reference, rather than accumulating the drive's
    # gravity/tracking error by re-anchoring every target at measured q. The
    # previous action is already part of the unchanged policy observation.
    previous = np.asarray(query['snapshot']['last_action'], dtype=np.float64)
    if previous.shape != (10,) or not np.isfinite(previous).all():
        raise ValueError('Expert needs the finite previous 10-dimensional action')
    previous_target = np.clip(PARK + 1.5*previous[3:], ARM_TARGET_LOWER, ARM_TARGET_UPPER)
    desired = a + min(dt / duration, 1.) * delta
    desired[3:] = previous_target + min(dt / duration, 1.)*(b[3:]-previous_target)
    # Base actions are velocities, arm actions are absolute joint targets.
    velocity = delta / max(duration, dt)
    local = rotation.inv().apply(np.r_[velocity[:2], 0.])
    return np.r_[np.clip(local[:2] / .15, -1., 1.),
                 np.clip(velocity[2] / .4, -1., 1.),
                 (desired[3:] - PARK) / 1.5]
