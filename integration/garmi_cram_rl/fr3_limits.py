"""FR3 arm velocity bounds for this project's CRAM/Giskard integration.

Nominal joint-space limits from Franka's FR3 specification:
https://frankarobotics.github.io/docs/robot_specifications.html#limits-for-franka-research-3-fr3

These are model bounds, not commanded speeds. Real FR3 control also requires
position-dependent velocity limits; this module does not implement an FCI driver.
"""

FR3_JOINT_VELOCITY_LIMITS = (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)


def apply_fr3_arm_velocity_limits(world):
    """Override only the 14 arm joints in one already-created GARMI world.

    Resolve every joint before writing, so an incompatible model fails without
    partially updating it. Preserve position, acceleration, jerk and all other
    joints' bounds. Do not use the upstream tightening helper: it cannot raise
    an existing 0.2 rad/s bound.
    """
    joints = [
        (world.get_connection_by_name(f'{side}_fr3_joint{index}'), limit)
        for side in ('left', 'right')
        for index, limit in enumerate(FR3_JOINT_VELOCITY_LIMITS, 1)
    ]
    for joint, limit in joints:
        joint.raw_dof.limits.lower.velocity = -limit
        joint.raw_dof.limits.upper.velocity = limit


def make_fr3_giskard_world(environment, spawn_pose):
    """Build the project's execution world with the same bounds as CRAM.

    As in the original GARMI factory, spawn_pose comes from localization rather
    than a second offset in the model. Furniture is still merged after GARMI.
    """
    from cram_vrb_lab.control.giskard_world import build_world_config
    from cram_vrb_lab.robots.garmi.giskard_config import WorldWithGarmiConfig
    from cram_vrb_lab.robots.garmi.joints import load_patched_urdf

    class WorldWithFR3ArmLimits(WorldWithGarmiConfig):
        def setup_world(self):
            super().setup_world()
            apply_fr3_arm_velocity_limits(self.world)

    return build_world_config(
        WorldWithFR3ArmLimits, environment, urdf=load_patched_urdf()
    )
