"""Run with the pinned CRAM Python environment; no ROS server or GPU required."""
from dataclasses import dataclass, field, replace

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import no_execution, real_robot, simulated_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_node import ActionNode
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.world import World

from garmi_cram_rl.pick_up_action import (
    RLPickUpAction, RLPickUpContext, RLPickUpFailure, TaskStage, task_stage, mount_task_motion,
)


@dataclass
class RecordingRuntime:
    world: object
    calls: list = field(default_factory=list)
    failure: str | None = None

    def pick_up(self, body, arm, reference, timeout):
        self.calls.append(('pick', body, arm, reference, timeout))
        if self.failure:
            raise RuntimeError(self.failure)
        return {'success': True, 'reference': reference}

    def perform_stage(self, stage):
        self.calls.append(stage)


@pytest.fixture
def runtime_context():
    world = World()
    runtime = RecordingRuntime(world)
    context = RLPickUpContext.from_context(Context(world=world, robot=None), runtime)
    return runtime, context


def test_expanding_and_parsing_action_does_not_call_policy(runtime_context):
    runtime, context = runtime_context
    node = execute_single(RLPickUpAction(context.world.root), context=context)
    node.notify()
    executable = node.parse()
    assert executable.giskard_executables == []
    assert runtime.calls == []


def test_sequence_calls_policy_between_navigation_and_transport(runtime_context):
    runtime, context = runtime_context
    root = sequential([
        task_stage(TaskStage.NAVIGATE),
        RLPickUpAction(context.world.root, reference='bowl-r020', timeout=17.),
        task_stage(TaskStage.TRANSPORT),
    ], context=context)
    with real_robot:
        root.perform()
    assert runtime.calls == [TaskStage.NAVIGATE,
        ('pick', context.world.root, Arms.RIGHT, 'bowl-r020', 17.), TaskStage.TRANSPORT]
    actions = [n for n in root.descendants if isinstance(n, ActionNode)]
    assert actions
    assert all(n.status == LifeCycleValues.SUCCEEDED for n in actions)
    assert actions[0].result == {'success': True, 'reference': 'bowl-r020'}


def test_failure_marks_action_and_prevents_transport(runtime_context):
    runtime, context = runtime_context
    runtime.failure = 'dropped'
    root = sequential([RLPickUpAction(context.world.root),
                       task_stage(TaskStage.TRANSPORT)], context=context)
    with real_robot, pytest.raises(RLPickUpFailure):
        root.perform()
    assert len(runtime.calls) == 1
    assert root.status == LifeCycleValues.FAILED
    actions = [n for n in root.descendants if isinstance(n, ActionNode)]
    assert all(n.status == LifeCycleValues.FAILED for n in actions)


def test_no_execution_never_calls_live_skill(runtime_context):
    runtime, context = runtime_context
    with no_execution:
        execute_single(RLPickUpAction(context.world.root), context=context).perform()
    assert runtime.calls == []


def test_planning_simulation_cannot_call_live_skill(runtime_context):
    runtime, context = runtime_context
    with simulated_robot, pytest.raises(RLPickUpFailure):
        execute_single(RLPickUpAction(context.world.root), context=context).perform()
    assert runtime.calls == []


def test_copied_world_cannot_call_original_robot(runtime_context):
    runtime, context = runtime_context
    other = World()
    context = replace(context, world=other)
    with real_robot, pytest.raises(RLPickUpFailure):
        execute_single(RLPickUpAction(other.root), context=context).perform()
    assert runtime.calls == []


def test_wrong_arm_never_calls_policy(runtime_context):
    runtime, context = runtime_context
    with real_robot, pytest.raises(RLPickUpFailure):
        execute_single(RLPickUpAction(context.world.root, arm=Arms.LEFT), context=context).perform()
    assert runtime.calls == []


def test_existing_motion_plan_stays_under_active_task_stage(runtime_context):
    runtime, context = runtime_context
    stage = task_stage(TaskStage.NAVIGATE)
    root = sequential([stage, RLPickUpAction(context.world.root)], context=context)
    stage.status = LifeCycleValues.RUNNING
    nested = execute_single(RLPickUpAction(context.world.root), context=context)
    assert mount_task_motion(root.plan, nested)
    assert nested.plan is root.plan
    assert nested.parent is stage
    assert runtime.calls == []


def test_live_viewer_can_publish_action_lifecycle(runtime_context):
    from cramera.live.bridge import Bridge
    from cramera.live.visualization import LiveVisualization

    runtime, context = runtime_context
    bridge = Bridge()
    viewer = LiveVisualization(context.world, bridge=bridge)
    bridge.attach(context.world)
    root = sequential([RLPickUpAction(context.world.root)], context=context)
    root.plan.node_callbacks.append(viewer.plan_callback(root.plan))
    with real_robot:
        root.perform()
    assert len(runtime.calls) == 1
    assert root.status == LifeCycleValues.SUCCEEDED
