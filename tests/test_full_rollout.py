"""Integration checks for offline rollout state and collision-audit isolation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from garmi_rl.giskard_expert import ApproachExpert
from giskard_full_rollout import rollout


@pytest.fixture(scope='module')
def expert():
    return ApproachExpert()


def query():
    value = json.loads(Path(__file__).with_name('expert-query.json').read_text())
    value['approach_seconds'] = .25
    return value


def test_collision_subsampling_does_not_change_trajectory(expert):
    request = query()
    original = copy.deepcopy(request)
    coarse = rollout(expert, request, subdivisions=1)
    dense = rollout(expert, request, subdivisions=10)
    assert request == original
    assert coarse['ticks'] == dense['ticks'] == 5
    for a, b in zip(coarse['trajectory'], dense['trajectory']):
        assert np.allclose(a['arm'], b['arm'], atol=1.e-7)
        assert np.allclose(a['base_pose'], b['base_pose'], atol=1.e-7)
    assert np.linalg.norm(np.asarray(dense['trajectory'][-1]['arm'])-dense['initial']['arm']) > 1.e-5


def test_rollout_joint_steps_fit_execution_interface(expert):
    result = rollout(expert, query())
    joints = np.array([result['initial']['arm']]+[p['arm'] for p in result['trajectory']])
    assert np.isfinite(joints).all()
    assert np.max(np.abs(np.diff(joints, axis=0))) <= .05+1.e-6
    velocities = np.array([p['base_velocity'] for p in result['trajectory']])
    assert np.all(np.abs(velocities) <= np.array([.15, .15, .4])+1.e-6)
