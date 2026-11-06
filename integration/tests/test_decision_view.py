from types import SimpleNamespace

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.world import World
from garmi_cram_rl.pick_up_action import task_stage, TaskStage, SkillNode
from garmi_cram_rl.decision_view import DecisionTrace, DecisionBridge
from test_adaptive_pick_up import task


def test_rejected_route_remains_failed_after_successful_alternative():
    trace = DecisionTrace()
    trace.observe(dict(event='pickup_attempt_started', attempt=1, maximum=3), 'prepare')
    for goal, passed in [([.2, 6.25, 1.57], False), ([.2, 6.4, 1.57], True)]:
        trace.observe(dict(event='station_probe_started', goal=goal), 'prepare')
        trace.observe(dict(event='station_probe', goal=goal, passed=passed), 'prepare')
    assert trace.rows[(1, 'route-1')]['status'] == 'FAILED'
    assert trace.rows[(1, 'route-2')]['status'] == 'SUCCEEDED'


def test_retry_keeps_previous_failure_and_stops_running_rows():
    trace = DecisionTrace()
    trace.observe(dict(event='pickup_attempt_started', attempt=1, maximum=3), 'one')
    trace.observe(dict(event='pickup_attempt_failed', reason='miss'), 'one')
    trace.observe(dict(event='pickup_attempt_started', attempt=2, maximum=3), 'two')
    trace.observe(dict(event='pickup_recovery_blocked', reason='<img onerror=alert(1)>'), 'two')
    assert trace.rows[(1, 'grasp')]['status'] == 'FAILED'
    assert not any(r['status'] == 'RUNNING' for r in trace.rows.values())
    assert '<' not in trace.rows[(2, 'stop')]['label']
    assert trace.rows[(2, 'stop')]['parent'] == 'two'


def test_bridge_decorates_view_without_changing_execution_tree(tmp_path):
    world = World()
    context = Context(world=world, robot=SimpleNamespace())
    root = sequential([task_stage(TaskStage.PREPARE_ATTEMPT), task_stage(TaskStage.TRANSPORT)], context=context)
    stage = next(n for n in root.descendants if getattr(n, 'stage', None) == TaskStage.PREPARE_ATTEMPT)
    stage.status = LifeCycleValues.RUNNING
    before = list(root.plan.nodes)
    bridge = DecisionBridge(decision_archive=tmp_path / 'plan.json')
    bridge.begin_plan(root.plan)
    bridge.observe_decision(dict(event='pickup_attempt_started', attempt=1, maximum=3))
    bridge.observe_decision(dict(event='station_probe_started', goal=[.2,6.25,1.57]))
    pending = bridge.plan_state.to_payload()
    bridge.observe_decision(dict(event='station_probe', goal=[.2,6.25,1.57], passed=True,
                                solver_steps=91, path_length_m=.46))
    payload = bridge.plan_state.to_payload()
    assert len(root.plan.nodes) == len(before)
    assert all(a is b for a, b in zip(root.plan.nodes, before))
    assert stage.status == LifeCycleValues.RUNNING
    assert any(n['status'] == 'SUCCEEDED' and '91 solver steps' in n['label'] for n in payload['nodes'])
    assert payload['signature'] != pending['signature']
    assert (tmp_path / 'plan.json').is_file()
    bridge.snapshot_plan()
    assert bridge.plan_state.to_payload() == payload  # No duplicate trace nodes.


def test_actual_cram_query_keeps_attempt_and_action_visible(task):
    from coraplex.execution_environment import real_robot
    from cramera.live.visualization import BridgePlanCallback
    bridge = DecisionBridge()
    bridge.begin_plan(task.root.plan)
    task.root.plan.node_callbacks.append(BridgePlanCallback(plan=task.root.plan, bridge=bridge))
    task.demo.record = lambda event, **values: bridge.observe_decision(dict(event=event, **values))
    with real_robot:
        task.root.perform()
    nodes = bridge.plan_state.to_payload()['nodes']
    assert any(n['label'] == 'Attempt 1/3' and n['status'] == 'SUCCEEDED' for n in nodes)
    assert any(n['label'] == 'RLPickUpAction · r0' and n['status'] == 'SUCCEEDED' for n in nodes)
    assert any('Select grasp · r0' in n['label'] for n in nodes)
    assert task.calls[-1] == 'transport'


def test_hold_details_attach_to_visible_action_in_existing_viewer(tmp_path):
    from garmi_cram_rl.pick_up_action import RLPickUpAction
    world = World()
    root = sequential([RLPickUpAction(world.root)], context=Context(world=world, robot=None))
    root.notify()
    skill = next(n for n in root.plan.nodes if isinstance(n, SkillNode))
    skill.status = LifeCycleValues.RUNNING
    bridge = DecisionBridge()
    bridge.begin_plan(root.plan)
    bridge.observe_decision(dict(event='rl_grasp_requested', reference='r0'))
    bridge.observe_decision(dict(event='handover_hold_check', passed=True, measurements={'height':.12}))
    nodes = {n['id']: n for n in bridge.plan_state.to_payload()['nodes']}
    detail = next(n for n in nodes.values() if 'Grasp verified' in n['label'])
    assert nodes[detail['parent']]['kind'] == 'ActionNode'
    assert nodes[detail['parent']]['label'] == 'RLPickUpAction · r0'


def test_nested_attempt_grasp_details_belong_to_grasp_not_attempt(task, monkeypatch):
    from coraplex.execution_environment import real_robot
    from cramera.live.visualization import BridgePlanCallback
    from garmi_cram_rl.demo_runtime import DemoPickUpRuntime
    bridge = DecisionBridge()
    bridge.begin_plan(task.root.plan)
    task.root.plan.node_callbacks.append(BridgePlanCallback(plan=task.root.plan, bridge=bridge))
    task.demo.record = lambda event, **values: bridge.observe_decision(dict(event=event, **values))
    def grasp(*args):
        task.demo.record('rl_grasp_requested', reference='r0')
        task.demo.record('handover_hold_check', passed=True, measurements={'height':.12})
        return {'success':True}
    monkeypatch.setattr(DemoPickUpRuntime, 'pick_up', grasp)
    with real_robot:
        task.root.perform()
    nodes = {n['id']:n for n in bridge.plan_state.to_payload()['nodes']}
    detail = next(n for n in nodes.values() if 'Grasp verified' in n['label'])
    assert nodes[detail['parent']]['label'] == 'RLPickUpAction · r0'
