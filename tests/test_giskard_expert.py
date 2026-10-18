"""Check planning/physics frame agreement and tilted-object goal updates."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from garmi_rl.giskard_expert import ApproachExpert


@pytest.fixture(scope='module')
def expert():
    return ApproachExpert()


@pytest.fixture
def query():
    return json.loads(Path(__file__).with_name('expert-query.json').read_text())


def test_measured_hand_to_goal_matches_physics_observation(expert, query):
    diagnostic = expert.sync(query)
    hand = np.array(diagnostic['hand_pose'])
    goal_position = np.array(diagnostic['goal_position'])
    local = hand[:3, :3].T @ (goal_position-hand[:3, 3])
    assert np.allclose(local, query['snapshot']['observation'][30:33], atol=3.e-5)
    expected_angle = 2*np.arccos(np.clip(abs(query['snapshot']['observation'][33]), 0., 1.))
    assert abs(diagnostic['angle']-expected_angle) < 3.e-5
    assert np.allclose([joint.velocity for joint in expert.arm], query['snapshot']['arm_velocity'])
    assert sum(d.has_hardware_interface for d in expert.world.active_degrees_of_freedom) == 10


def test_goal_follows_full_object_rotation(expert, query):
    altered = copy.deepcopy(query)
    obj = np.array(altered['snapshot']['object_state'])
    obj[3:7] = (Rotation.from_euler('x', 4., degrees=True)*Rotation.from_quat(obj[[4, 5, 6, 3]])).as_quat()[[3, 0, 1, 2]]
    altered['snapshot']['object_state'] = obj.tolist()
    result = expert.sync(altered)
    ref = expert.references[query['reference']]
    expected = obj[:3]+Rotation.from_quat(obj[[4, 5, 6, 3]]).apply(ref['position'])
    assert np.allclose(result['goal_position'], expected, atol=1.e-8)


def test_qp_returns_finite_bounded_command(expert, query):
    result = expert.command(query)
    assert result['status'] == 'command_found'
    assert np.isfinite(result['action']).all()
    assert np.max(np.abs(result['action'][:3])) <= 1.


def test_reused_solver_does_not_keep_another_query_goal(expert, query):
    first = expert.command(query)
    other = copy.deepcopy(query)
    other['snapshot']['object_state'][0] += .1
    expert.command(other)
    repeated = expert.command(query)
    assert np.allclose(first['action'], repeated['action'], atol=1.e-5)
    assert repeated['compile_seconds'] == 0.


def test_only_fingers_may_touch_bowl(expert, query):
    expert.command(query)
    matrix = expert.world.collision_manager.collision_matrix
    pairs = {(c.body_a.name.name, c.body_b.name.name) for c in matrix.collision_checks}
    assert any('target_bowl' in pair and 'right_fr3_link7' in pair for pair in pairs)
    assert not any('target_bowl' in pair and any('finger' in n for n in pair) for pair in pairs)
    hand = expert.world.get_body_by_name('right_fr3_link7')
    assert expert.world.collision_manager.get_buffer_zone_distance(hand, expert.bowl) == .001


def test_pregrasp_is_offset_along_hand_approach_axis(expert, query):
    query['approach_stage'] = 'pregrasp'
    result = expert.sync(query)
    rotation = Rotation.from_quat(np.array(result['goal_quaternion'])[[1, 2, 3, 0]])
    expected = np.array(result['goal_position'])-rotation.apply([0., 0., .08])
    assert result['stage'] == 'pregrasp'
    assert np.allclose(result['command_goal_position'], expected)
    # Once the caller has reached the pre-pose, a deviation must not send it back.
    query['approach_stage'] = 'approach_grasp'
    result = expert.sync(query)
    assert result['stage'] == 'approach_grasp'
    assert np.allclose(result['command_goal_position'], result['goal_position'])


def test_perturbation_does_not_relabel_grasp_or_leak_to_next_query(expert, query):
    original=expert.sync(query)
    noisy=copy.deepcopy(query)
    noisy['perturbation_pose']=[.6,.2,1.1,1.,0.,0.,0.]
    changed=expert.sync(noisy)
    assert changed['perturbation_command']
    assert np.allclose(changed['command_goal_position'],noisy['perturbation_pose'][:3])
    assert np.allclose(changed['goal_position'],original['goal_position'])
    assert np.allclose(changed['hand_pose'],original['hand_pose'])
    repeated=expert.sync(query)
    assert not repeated['perturbation_command']
    assert np.allclose(repeated['command_goal_position'],original['command_goal_position'])
