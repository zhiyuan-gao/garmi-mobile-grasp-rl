"""CRAM action for the existing live mobile-grasp skill.

The execution boundary is essential: plan expansion, rendering and reachability
probes must never send policy commands. No CRAM core modifications are needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import Protocol

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.plans.executables import Executable, GiskardExecutable
from coraplex.plans.factories import execute_single
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import ActionNode, ExecutionBoundaryNode
from coraplex.robot_plans.actions.base import ActionDescription
from giskardpy.motion_statechart.data_types import LifeCycleValues
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


class TaskStage(str, Enum):
    RESOLVE_PICKUP = 'resolve_pickup'
    PREPARE_ATTEMPT = 'prepare_attempt'
    PREPARE = 'prepare'
    NAVIGATE = 'navigate'
    TRANSPORT = 'transport'
    RETREAT = 'retreat'


class PickUpRuntime(Protocol):
    world: World

    def pick_up(self, body: Body, arm: Arms, reference: str | None, timeout: float) -> dict: ...

    def perform_stage(self, stage: TaskStage) -> None: ...


@dataclass
class RLPickUpFailure(PlanFailure):
    reason: str

    def error_message(self) -> str:
        return f'RL pick-up failed: {self.reason}'


@dataclass
class TaskStageFailure(PlanFailure):
    stage: TaskStage
    reason: str

    def error_message(self) -> str:
        return f'{self.stage.value} failed: {self.reason}'


@dataclass(eq=False)
class RLPickUpContext(Context):
    """Bind a live skill to one world, outside serializable action parameters."""

    pickup_runtime: PickUpRuntime = field(kw_only=True, repr=False)

    @classmethod
    def from_context(cls, context: Context, runtime: PickUpRuntime) -> RLPickUpContext:
        values = {f.name: getattr(context, f.name) for f in fields(Context)
                  if f.init and f.name != 'plan'}
        return cls(**values, pickup_runtime=runtime)


@dataclass
class RLPickUpAction(ActionDescription):
    """Take hold of an object and lift it using RL plus the fixed finish.

    Navigation and starting-posture preparation belong to the enclosing plan.
    ``reference=None`` delegates selection to the existing position-mask rule.
    The runtime verifies physical success before updating the semantic grasp.
    """

    object_designator: Body
    arm: Arms = Arms.RIGHT
    reference: str | None = None
    timeout: float = 180.

    @property
    def _action_plan(self):
        return execute_single(SkillNode(action=self))


@dataclass(eq=False, repr=False)
class SkillNode(ExecutionBoundaryNode):
    action: RLPickUpAction | None = field(default=None, kw_only=True)
    stage: TaskStage | None = field(default=None, kw_only=True)

    def notify(self) -> None:
        # Expansion is deliberately free of execution side effects.
        pass

    def parse(self) -> SkillExecutable:
        return SkillExecutable(context=self.context, node=self)

    def __node_label__(self):
        return 'RL grasp execution' if self.action is not None else self.stage.value


@dataclass
class SkillExecutable(Executable):
    node: SkillNode = field(kw_only=True)

    def execute(self):
        if GiskardExecutable.execution_type == ExecutionType.NO_EXECUTION:
            return None
        # Nested action nodes are parsed directly by CRAM. Emit their lifecycle
        # here too, so a policy attempt appears in the live plan and recordings.
        nodes = [n for n in reversed(self.node.path)
                 if isinstance(n, ActionNode) and isinstance(n.action, RLPickUpAction)
                 and n.status == LifeCycleValues.NOT_STARTED]
        nodes.append(self.node)
        for node in nodes:
            node.status = LifeCycleValues.RUNNING
            node.start_time = datetime.now()
            node.plan.notify_node_started(node)
        try:
            result = self.invoke()
        except Exception as error:
            if isinstance(error, PlanFailure):
                failure = error
            elif self.node.action is not None:
                failure = RLPickUpFailure(str(error))
            else:
                failure = TaskStageFailure(self.node.stage, str(error))
            for node in nodes:
                node.status, node.reason = LifeCycleValues.FAILED, failure
            if failure is error:
                raise
            raise failure from error
        else:
            for node in nodes:
                node.status, node.result = LifeCycleValues.SUCCEEDED, result
            return result
        finally:
            for node in reversed(nodes):
                node.end_time = datetime.now()
                node.plan.notify_node_ended(node)

    def invoke(self):
        if GiskardExecutable.execution_type != ExecutionType.REAL:
            raise RLPickUpFailure('Live RL execution is unavailable in a planning-world simulation')
        if not isinstance(self.context, RLPickUpContext):
            raise RLPickUpFailure('No RL runtime is bound to this CRAM context')
        runtime = self.context.pickup_runtime
        if runtime.world is not self.context.world:
            raise RLPickUpFailure('A copied planning world cannot command the live robot')
        action = self.node.action
        if action is None:
            return runtime.perform_stage(self.node.stage)
        if action.arm != Arms.RIGHT:
            raise RLPickUpFailure('The deployed policy supports the right arm only')
        if action.timeout <= 0:
            raise RLPickUpFailure('The execution timeout must be positive')
        return runtime.pick_up(action.object_designator, action.arm, action.reference, action.timeout)


def task_stage(stage: TaskStage) -> SkillNode:
    """Defer a validated demo stage until its turn in the CRAM sequence."""
    return SkillNode(stage=stage)


def mount_task_motion(task_plan, motion) -> bool:
    """Keep existing navigation/place subplans inside the active task's viewer tree."""
    if task_plan is None:
        return False
    running=[node for node in task_plan.nodes if isinstance(node, SkillNode)
             and node.stage is not None and node.status == LifeCycleValues.RUNNING]
    if not running:
        return False
    max(running,key=lambda node:len(node.path)).mount_subplan(motion)
    return True
