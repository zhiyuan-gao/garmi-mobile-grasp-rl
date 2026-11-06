from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from coraplex.datastructures.enums import Arms
from garmi_cram_rl.demo_runtime import DemoPickUpRuntime
from garmi_cram_rl.pick_up_action import RLPickUpFailure


@pytest.fixture
def demo():
    events = []
    root, tool = object(), object()
    body = SimpleNamespace(name='bowl', parent_kinematic_structure_entity=root)
    world = SimpleNamespace(root=root, move_branch=Mock(side_effect=lambda *args: events.append('attach')))
    result = {'result': {'success': True}, 'request_id': 'attempt',
              'snapshot': {'reference': 'r020'}}
    demo = SimpleNamespace(world=world, body=body, tool=tool, events=events,
        rl=SimpleNamespace(snapshot=Mock(return_value={'owner': 'cram'}),
                           grasp=Mock(return_value=result)),
        prepare_grasp_view=Mock(),
        verify_hold=Mock(side_effect=lambda: events.append('verify') or {'height': .1}),
        sync=SimpleNamespace(pull=Mock(side_effect=lambda world: events.append('sync'))),
        object_pose=Mock(return_value=[0, 0, 1]), record=Mock())
    return demo


def test_physical_verification_precedes_semantic_attachment(demo):
    result = DemoPickUpRuntime(demo).pick_up(demo.body, Arms.RIGHT, 'r020', 12.)
    assert demo.events == ['verify', 'sync', 'attach']
    demo.rl.grasp.assert_called_once_with('r020', timeout=12.)
    demo.world.move_branch.assert_called_once_with(demo.body, demo.tool)
    assert result['measurements'] == {'height': .1}
    assert result['request_id'] == 'attempt'
    assert result['reference'] == 'r020'


def test_failed_hold_never_asserts_semantic_grasp(demo):
    demo.verify_hold.side_effect = RuntimeError('dropped after handback')
    with pytest.raises(RuntimeError):
        DemoPickUpRuntime(demo).pick_up(demo.body, Arms.RIGHT, None, 12.)
    demo.world.move_branch.assert_not_called()
    demo.record.assert_called_once_with('rl_grasp_requested', reference=None)


@pytest.mark.parametrize('case', ['wrong_object', 'left_arm', 'already_held', 'busy'])
def test_invalid_request_does_not_start_policy(demo, case):
    body, arm = demo.body, Arms.RIGHT
    if case == 'wrong_object':
        body = object()
    if case == 'left_arm':
        arm = Arms.LEFT
    if case == 'already_held':
        demo.body.parent_kinematic_structure_entity = demo.tool
    if case == 'busy':
        demo.rl.snapshot.return_value = {'owner': 'rl'}
    with pytest.raises(RLPickUpFailure):
        DemoPickUpRuntime(demo).pick_up(body, arm, None, 12.)
    demo.rl.grasp.assert_not_called()
    demo.world.move_branch.assert_not_called()
