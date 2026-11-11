"""Check limits on actual GARMI worlds, including copied planning worlds."""
from copy import deepcopy

from cram_vrb_lab.robots.garmi.giskard_config import WorldWithGarmiConfig
from cram_vrb_lab.robots.garmi.joints import load_patched_urdf
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from garmi_cram_rl.fr3_limits import (
    apply_fr3_arm_velocity_limits,
    make_fr3_giskard_world,
)


EXPECTED = (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26)
ARM_LIMITS = {
    f'{side}_fr3_joint{i}': v
    for side in ('left', 'right')
    for i, v in enumerate(EXPECTED, 1)
}


def build(config):
    with config.world.modify_world():
        config.setup_world()
    return config.world


def limits(world):
    result = {}
    for joint in world.get_connections_by_type(ActiveConnection1DOF):
        bound = joint.raw_dof.limits
        result[str(joint.name)] = tuple(
            getattr(side, derivative)
            for side in (bound.lower, bound.upper)
            for derivative in ('position', 'velocity', 'acceleration', 'jerk')
        )
    return result


def assert_arms(world):
    for name, velocity in ARM_LIMITS.items():
        bound = world.get_connection_by_name(name).raw_dof.limits
        assert bound.lower.velocity == -velocity
        assert bound.upper.velocity == velocity


def test_cram_override_changes_only_arm_velocity_bounds_and_survives_copy():
    world = build(WorldWithGarmiConfig(urdf=load_patched_urdf()))
    before = limits(world)
    apply_fr3_arm_velocity_limits(world)
    assert_arms(world)
    after = limits(world)
    assert before.keys() == after.keys()
    arm_names = {str(world.get_connection_by_name(n).name) for n in ARM_LIMITS}
    for name, old in before.items():
        if name in arm_names:
            assert tuple(v for i, v in enumerate(old) if i not in (1, 5)) == tuple(
                v for i, v in enumerate(after[name]) if i not in (1, 5)
            )
        else:
            assert after[name] == old
    assert_arms(deepcopy(world))
    apply_fr3_arm_velocity_limits(world)
    assert limits(world) == after


def test_execution_factory_and_upstream_default_remain_independent():
    execution = build(make_fr3_giskard_world(None, None))
    assert_arms(execution)
    original = build(WorldWithGarmiConfig(urdf=load_patched_urdf()))
    for name in ARM_LIMITS:
        bound = original.get_connection_by_name(name).raw_dof.limits
        assert bound.lower.velocity == -0.2
        assert bound.upper.velocity == 0.2
