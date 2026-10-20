"""Protect strict quota accounting, split assignment and initial-state isolation."""
import collections
import copy

import numpy as np
import pytest

from garmi_rl.expert_collection import LocalCaseSampler, position_key, strict_success


def specification():
    case = dict(object='mug', reference='a', robot_state=[0., .05, .026, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                object_state=[.91, .14, .947, .70710678, 0., 0., -.70710678, 0., 0., 0., 0., 0., 0.])
    other = copy.deepcopy(case)
    other['reference'] = 'b'
    return dict(object='mug', xy_radius_m=.004, base_bounds_task_xy=[[-.14, -.54], [.14, .54]],
                references={'a': [dict(id='a-window', case=case)], 'b': [dict(id='b-window', case=other)]})


def successful_row():
    return dict(reason='success', handover_trigger=1, handover_distance=.002,
                handover_angle=.02, close_start_distance=.004, close_start_angle=.04,
                height=.12, contact=True, supported=False, slip=.005, hold_seconds=2.05,
                collision_bodies={})


@pytest.mark.parametrize('field,value', [('close_start_distance', .00508), ('handover_trigger', 2),
    ('close_start_angle', .05001), ('height', .079), ('slip', .02), ('hold_seconds', 1.99),
    ('collision_bodies', {'right_fr3_link6': .2}), ('close_start_distance', float('nan'))])
def test_physical_success_with_bad_strict_metrics_cannot_fill_a_quota(field, value):
    row = successful_row()
    assert strict_success(row)
    row[field] = value
    assert not strict_success(row)


def test_legacy_success_without_measured_close_start_is_not_strict():
    row = successful_row()
    del row['close_start_distance']
    assert not strict_success(row)
    assert not strict_success(dict(reason='expert_query_failed'))


def test_sampler_only_fills_missing_reference_and_split():
    sampler = LocalCaseSampler(specification(), 10)
    accepted = {'train': {'a': 1, 'b': 0}, 'validation': {'a': 0, 'b': 1}}
    cases = sampler.sample(100, ['a', 'b'], accepted, {'train': 1, 'validation': 1})
    assert collections.Counter((c['split'], c['reference']) for c in cases) == {
        ('train', 'b'): 4, ('validation', 'a'): 4}
    assert sampler.sample(10, ['a'], {'train': {'a': 1}, 'validation': {'a': 1}},
                          {'train': 1, 'validation': 1}) == []


def test_sampler_is_reproducible_separated_from_heldout_and_keeps_verified_windows():
    spec = specification()
    heldout = [spec['references']['a'][0]['case']]
    first = LocalCaseSampler(spec, 20, heldout)
    second = LocalCaseSampler(spec, 20, heldout)
    accepted = {'train': {}, 'validation': {}}
    quotas = {'train': 200, 'validation': 20}
    a = first.sample(300, ['a', 'b'], accepted, quotas)
    assert a == second.sample(300, ['a', 'b'], accepted, quotas)
    xy = np.array([position_key(c) for c in heldout+a])
    distances = np.max(np.abs(xy[:, None]-xy[None, :]), axis=2)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() > .0002
    assert np.max(np.abs(xy[1:]-xy[0])) <= .004
    for case in a:
        original = spec['references'][case['reference']][0]['case']
        assert case['object_state'][2:] == original['object_state'][2:]
        assert case['robot_state'][2:] == original['robot_state'][2:]
