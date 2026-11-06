import math
from types import SimpleNamespace

import pytest

from garmi_cram_rl.skill_selection import station_candidates, choose_reference, recovery_rejection


@pytest.fixture
def description():
    return dict(eligible=True, rejection=None, allowed_references=['r0', 'r1', 'r2'],
        skill=dict(base_bounds_map_xy=[[-.25, 6.2], [.85, 6.5]], base_yaw=math.pi/2),
        snapshot=dict(base_map_position=[0.,5.8,.0259], object_map_position=[.2,7.4,.978],
            object_velocity=[0.]*6, grasp_measurements=dict(supported=1.,contact=0.,collision=0.)))


def test_station_candidates_follow_object_and_stay_inside_training_region(description):
    first = station_candidates(description)
    description['snapshot']['object_map_position'][0] = .5
    second = station_candidates(description)
    assert first[0][0] == .2
    assert second[0][0] == .5
    for x,y,yaw in first+second:
        assert -.25 < x < .85 and 6.2 < y < 6.5
        assert yaw == math.pi/2


def test_requested_station_outside_region_is_rejected(description):
    with pytest.raises(ValueError): station_candidates(description, preferred=[2,6.3,math.pi/2])


def test_reference_selection_excludes_attempted_and_rechecks_current_mask(description):
    assert choose_reference(description, ['r0']) == 'r1'
    description['allowed_references'] = ['r2']
    assert choose_reference(description, ['r0']) == 'r2'
    with pytest.raises(ValueError): choose_reference(description, ['r2'])


@pytest.mark.parametrize('cause', ['collision', 'dropped', 'client_timeout', 'cancelled'])
def test_unsafe_failure_reason_cannot_be_released(description, cause):
    state=dict(owner='holding', result={'reason':cause, 'success':False})
    assert recovery_rejection(state, description) is not None


def test_only_empty_supported_stationary_miss_can_recover(description):
    state=dict(owner='holding', result={'reason':'attempt_failed','success':False})
    assert recovery_rejection(state, description) is None
    description['snapshot']['grasp_measurements']['contact'] = 1.
    assert recovery_rejection(state, description) is not None
    description['snapshot']['grasp_measurements']['contact'] = 0.
    description['eligible'] = False
    assert recovery_rejection(state, description) is not None
