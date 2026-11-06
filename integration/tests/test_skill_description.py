from types import SimpleNamespace
from unittest.mock import Mock
import math

import numpy as np
import pytest
from garmi_cram_rl.skill_description import describe_skill


@pytest.fixture
def backend():
    flags=SimpleNamespace(cpu=lambda:SimpleNamespace(numpy=lambda:np.array([True,False,True])))
    mask=SimpleNamespace(data={'object_name':'bowl','reference_ids':['r0','r1','r2'],
        'bowl_origin_z_m':.978,'regions':[{'bounds_map_xy':[[-.23,7.23],[.55,7.63]]}]},
        allowed=Mock(return_value=[flags]))
    snapshot=dict(base_map_position=[0.,5.8,.0259],object_map_position=[.2,7.4,.978],
        base_map_yaw=math.pi/2,object_map_euler=[0.,0.,0.],object_velocity=[0.]*6,
        arm_position=[0.,-.785398,0.,-2.356194,0.,1.570796,.785398],
        base_velocity=[0.]*3,joint_velocity=[0.]*7)
    values={k:np.array([v]) for k,v in dict(height=0.,slip=0.,contact=0.,supported=1.,collision=0.).items()}
    return SimpleNamespace(snapshot=lambda:dict(snapshot),source=snapshot,
        mask=mask,references={'r0':{},'r1':{},'r2':{}},reference=None,
        bundle=SimpleNamespace(manifest={'checkpoint_sha256':'model','action_contract':'base_velocity_joint_delta_v1'}),
        task={'kitchen':{'base_bounds_map_xy':[[-.25,6.2],[.85,6.5]],'base_map_yaw_degrees':90.}},
        env=None,mdp=SimpleNamespace(diagnostics=lambda env:values))


def test_metadata_query_does_not_require_already_navigating_into_region(backend):
    info=describe_skill(backend)
    assert info['eligible']
    assert info['allowed_references']==['r0','r2']
    assert info['admission'] is not None
    assert backend.reference is None
    assert 'grasp_measurements' not in backend.source


def test_query_reuses_pose_restrictions_from_admission(backend):
    backend.source['object_map_euler']=[.02,0.,0.]
    info=describe_skill(backend)
    assert not info['eligible']
    assert info['allowed_references']==[]
    backend.mask.allowed.assert_not_called()


def test_query_reports_ready_after_valid_navigation(backend):
    backend.source['base_map_position']=[.2,6.3,.0259]
    assert describe_skill(backend)['admission'] is None
