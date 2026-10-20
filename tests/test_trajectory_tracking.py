import numpy as np
import pytest
from garmi_rl.trajectory_tracking import base_tracking_velocity, retime_plan


def test_world_motion_is_rotated_into_actual_base_axes():
    result = base_tracking_velocity([0,0,np.pi/2], [.005,0,np.pi/2], [0,0,np.pi/2], .05, .5)
    assert np.allclose(result, [0,-.1,0])


def test_finished_trajectory_corrects_error_then_stops():
    pose = [0,0,0]
    assert np.allclose(base_tracking_velocity(pose,pose,[-.01,.02,.03],.05,.5),
                       [.02*np.cos(.03)-.04*np.sin(.03), -.02*np.sin(.03)-.04*np.cos(.03), -.06])
    assert np.allclose(base_tracking_velocity(pose,pose,pose,.05,.5), 0)


def test_yaw_wrap_does_not_request_a_full_turn():
    value = base_tracking_velocity([0,0,np.pi-.01],[0,0,-np.pi+.01], [0,0,np.pi-.01],.1,.5)
    assert np.allclose(value,[0,0,.2])


def test_invalid_period_rejected():
    with pytest.raises(ValueError):
        base_tracking_velocity([0,0,0],[0,0,0],[0,0,0],0,.5)


def test_retiming_preserves_path_and_respects_start_stop_acceleration():
    def state(x):
        return dict(base_pose=[x,0,0],arm=[x]*7,hand=np.eye(4).tolist())
    plan = dict(initial=state(0),trajectory=[state(.005),state(.01)],ticks=2)
    slow = retime_plan(plan,.05,[.15,.15,.4],[.3,.3,.8])
    assert slow['retiming']['factor'] > 1
    assert np.all(np.array(slow['retiming']['peak_base_acceleration'])<=np.array([.3,.3,.8])+1.e-9)
    assert slow['trajectory'][-1]['base_pose'] == plan['trajectory'][-1]['base_pose']
    assert slow['trajectory'][-1]['arm'] == plan['trajectory'][-1]['arm']
    for i, original in enumerate(plan['trajectory']):
        actual=slow['trajectory'][(i+1)*slow['retiming']['factor']-1]
        assert np.allclose(actual['base_pose'],original['base_pose'])
    assert len(plan['trajectory']) == 2
    plan['initial']['base_velocity']=[.1,0,0]
    with pytest.raises(ValueError,match='stationary'):
        retime_plan(plan,.05,[.15,.15,.4],[.3,.3,.8])
