import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from garmi_cram_rl.contract import MapFrame, validate_start
from garmi_cram_rl.session import Session


def test_frames_match_training_kitchen_and_preserve_orientation():
    frame = MapFrame()
    np.testing.assert_allclose(frame.position_to_map([0., 0., .0259]), [.3, 6.35, .0259])
    point = [.2, 7.4, .978]
    np.testing.assert_allclose(frame.position_to_map(frame.position_to_task(point)), point)
    np.testing.assert_allclose(frame.quaternion_to_map([1, 0, 0, 0]), [math.sqrt(.5), 0, 0, math.sqrt(.5)])
    np.testing.assert_allclose(frame.quaternion_to_task([1, 0, 0, 0]), [math.sqrt(.5), 0, 0, -math.sqrt(.5)])


@pytest.fixture
def backend():
    class Backend:
        def __init__(self):
            self.events = []
            self.reason = None
        def admit(self, reference):
            if reference == 'blocked':
                raise ValueError('blocked')
        def begin(self, reference): self.events.append('begin')
        def step_policy(self):
            self.events.append('step')
            return self.reason
        def hold(self): self.events.append('hold')
        def return_to_cram(self): self.events.append('release')
    return Backend()


def test_exclusive_ownership_and_duplicate_requests_do_not_reset(backend):
    session = Session(backend)
    session.start('one', 'ref', 0.)
    session.start('one', 'ref', .1)
    with pytest.raises(RuntimeError): session.start('two', 'ref', .2)
    with pytest.raises(RuntimeError): session.release('one')
    assert backend.events == ['begin']
    assert session.owner == 'rl'


def test_success_holds_without_reset_then_explicitly_returns_control(backend):
    session = Session(backend)
    session.start('one', 'ref', 0.)
    backend.reason = 'success'
    session.tick(.05)
    session.tick(.10)
    assert session.result == {'success': True, 'reason': 'success'}
    assert backend.events == ['begin', 'step', 'hold']
    assert session.owner == 'holding'
    session.release('one')
    assert session.owner == 'cram'
    assert backend.events[-1] == 'release'
    session.start('one', 'ref', .2)
    assert backend.events == ['begin', 'step', 'hold', 'release']


@pytest.mark.parametrize('reason', ['collision', 'dropped', 'attempt_failed', 'out_of_bounds'])
def test_failure_is_never_reported_as_success(backend, reason):
    session = Session(backend)
    session.start('one', 'ref', 0.)
    backend.reason = reason
    session.tick(.05)
    assert session.result == {'success': False, 'reason': reason}


def test_disconnected_client_stops_without_running_a_new_action(backend):
    session = Session(backend)
    session.start('one', 'ref', 0.)
    session.tick(2.01)
    assert backend.events == ['begin', 'hold']
    assert session.result == {'success': False, 'reason': 'client_timeout'}


def test_rejected_start_leaves_classical_controller_in_charge(backend):
    session = Session(backend)
    with pytest.raises(ValueError): session.start('one', 'blocked', 0.)
    assert session.owner == 'cram'
    assert backend.events == []


def test_masked_grasp_rejected_without_changing_pose():
    snapshot = dict(base_map_position=[.3, 6.35, .0259], object_map_position=[.2, 7.4, .978],
        arm_position=[0., -.785398, 0., -2.356194, 0., 1.570796, .785398],
        base_velocity=[0., 0., 0.], joint_velocity=[0.] * 7,
        base_map_yaw=math.pi / 2, object_map_euler=[0., 0., 0.])
    task = {'kitchen': {'base_bounds_map_xy': [[-.25, 6.2], [.85, 6.5]]}}
    mask = SimpleNamespace(data={'bowl_origin_z_m': .978, 'reference_ids': ['good', 'blocked']},
                           allowed=lambda *args: torch.tensor([[True, False]]))
    assert validate_start(snapshot, task, mask) == 'good'
    snapshot.update(joint_velocity=[.04] * 7, joint_settling_range=[.000003] * 7,
                    settling_window_seconds=.5)
    assert validate_start(snapshot, task, mask) == 'good'
    snapshot['joint_settling_range'][2] = .01
    with pytest.raises(ValueError, match='settle'):
        validate_start(snapshot, task, mask)
    snapshot['joint_settling_range'][2] = .000003
    with pytest.raises(ValueError): validate_start(snapshot, task, mask, 'blocked')
    snapshot['base_map_position'][0] = 1.
    with pytest.raises(ValueError): validate_start(snapshot, task, mask, 'good')
