from types import SimpleNamespace
from unittest.mock import Mock
import math

import pytest
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import sequential
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.world import World
from garmi_cram_rl.pick_up_action import RLPickUpContext, RLPickUpFailure, TaskStage, task_stage
from garmi_cram_rl.demo_runtime import DemoPickUpRuntime
from garmi_cram_rl.adaptive_pick_up import AdaptivePickUpRuntime, RLPickUpAttempt
from garmi_cram_rl.session import Session


@pytest.fixture
def task(monkeypatch):
    world=World(); calls=[]
    description=dict(eligible=True,rejection=None,admission=None,allowed_references=['r0','r1','r2'],
        skill=dict(object_name='bowl',arm='RIGHT',checkpoint_sha256='model',
                   base_bounds_map_xy=[[-.25,6.2],[.85,6.5]],base_yaw=math.pi/2),
        snapshot=dict(base_map_position=[0.,5.8,0.],object_map_position=[.2,7.4,.978],
                      object_velocity=[0.]*6,grasp_measurements=dict(supported=1,contact=0,collision=0)))
    state=dict(owner='cram',request_id='one',result=None)
    def recover(request):
        calls.append('recover');state['owner']='cram'
    demo=SimpleNamespace(world=world,body=world.root,
        options=SimpleNamespace(base=None,reference=None,max_grasp_attempts=3),
        rl=SimpleNamespace(describe_skill=lambda:description,snapshot=lambda:state,recover=recover),
        record=Mock(), training_posture=lambda label:calls.append('park'),
        probe_navigation=Mock(return_value=True),probe_preparation=Mock(return_value=True),
        open_gripper_verified=lambda:calls.append('open'),
        navigate_to_grasp_region=lambda pose:calls.append(('navigate',pose)),
        transport=lambda:calls.append('transport'))
    runtime=AdaptivePickUpRuntime(demo)
    demo.context=RLPickUpContext.from_context(Context(world=world,robot=None),runtime)
    root=sequential([task_stage(TaskStage.RESOLVE_PICKUP),task_stage(TaskStage.TRANSPORT)],context=demo.context)
    demo.task_plan=root.plan
    def grasp(self,body,arm,reference,timeout):
        calls.append(('grasp',reference))
        return {'success':True}
    monkeypatch.setattr(DemoPickUpRuntime,'pick_up',grasp)
    return SimpleNamespace(root=root,demo=demo,runtime=runtime,calls=calls,state=state,description=description)


def test_cram_query_stops_after_first_success(task):
    with real_robot:task.root.perform()
    assert [c for c in task.calls if isinstance(c,tuple) and c[0]=='grasp'] == [('grasp','r0')]
    assert task.calls[-1]=='transport'
    assert task.runtime.attempt_number==1


def test_supported_miss_recovers_and_selects_new_reference(task,monkeypatch):
    def grasp(self,body,arm,reference,timeout):
        task.calls.append(('grasp',reference))
        if reference=='r0':
            task.state.update(owner='holding',result={'success':False,'reason':'attempt_failed'})
            # The current position filter can change after a failed approach.
            task.description['allowed_references']=['r0','r2']
            raise RuntimeError('miss')
        return {'success':True}
    monkeypatch.setattr(DemoPickUpRuntime,'pick_up',grasp)
    with real_robot: task.root.perform()
    assert task.runtime.attempted_references==['r0','r2']
    assert task.calls.index('recover') < task.calls.index('open') < task.calls.index(('grasp','r2'))
    assert task.calls[-1]=='transport'
    task.demo.probe_preparation.assert_called_once_with(open_gripper=True)


@pytest.mark.parametrize('reason',['collision','dropped','client_timeout'])
def test_unsafe_failure_stops_without_recovery_or_transport(task,monkeypatch,reason):
    def grasp(*args):
        task.state.update(owner='holding',result={'success':False,'reason':reason})
        raise RuntimeError(reason)
    monkeypatch.setattr(DemoPickUpRuntime,'pick_up',grasp)
    with real_robot,pytest.raises(RLPickUpFailure):task.root.perform()
    assert task.runtime.attempted_references==['r0']
    assert 'recover' not in task.calls and 'transport' not in task.calls
    assert task.root.status==LifeCycleValues.FAILED


def test_attempt_budget_is_enforced(task,monkeypatch):
    def grasp(*args):
        task.state.update(owner='holding',result={'success':False,'reason':'attempt_failed'})
        raise RuntimeError('miss')
    monkeypatch.setattr(DemoPickUpRuntime,'pick_up',grasp)
    with real_robot,pytest.raises(RLPickUpFailure):task.root.perform()
    assert task.runtime.attempted_references==['r0','r1','r2']
    assert task.calls.count('recover')==2
    assert 'transport' not in task.calls


def test_tilted_or_out_of_region_object_stops_before_motion(task):
    task.description.update(eligible=False,rejection='outside validated pose range')
    with real_robot,pytest.raises(RLPickUpFailure):task.root.perform()
    assert task.calls==[]


def test_server_revalidates_before_releasing_failed_controller(task):
    backend=SimpleNamespace(describe_skill=lambda:task.description,return_to_cram=Mock())
    session=Session(backend,owner='holding',request_id='one',result={'success':False,'reason':'attempt_failed'})
    task.description['snapshot']['grasp_measurements']['contact']=1
    with pytest.raises(ValueError):session.recover('one')
    backend.return_to_cram.assert_not_called()
    assert session.owner=='holding'
    task.description['snapshot']['grasp_measurements']['contact']=0
    session.recover('one')
    backend.return_to_cram.assert_called_once()
    assert session.owner=='cram'


def test_mug_task_accepts_only_mug_skill(task):
    task.demo.skill_object_name = 'mug'
    task.description['skill']['object_name'] = 'mug'
    with real_robot:
        task.root.perform()
    assert ('grasp', 'r0') in task.calls


def test_mug_task_rejects_bowl_skill_before_motion(task):
    task.demo.skill_object_name = 'mug'
    with real_robot, pytest.raises(RLPickUpFailure, match='mug/right-arm'):
        task.root.perform()
    assert task.calls == []
