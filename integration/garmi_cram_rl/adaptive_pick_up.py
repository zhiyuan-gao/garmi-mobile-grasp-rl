"""Ground a bounded set of CRAM pick-up attempts using measured skill applicability."""
from dataclasses import dataclass, field
import time

from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.failures import EmptyUnderspecified, RECOVERABLE_FAILURES
from coraplex.plans.executables import UnderspecifiedExecutable
from coraplex.plans.underspecified import UnderspecifiedNode
from coraplex.robot_plans.actions.base import ActionDescription
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.world_description.world_entity import Body

from garmi_cram_rl.demo_runtime import DemoPickUpRuntime
from garmi_cram_rl.pick_up_action import RLPickUpAction, RLPickUpFailure, TaskStage, task_stage, mount_task_motion
from garmi_cram_rl.skill_selection import choose_reference, station_candidates, recovery_rejection


@dataclass
class RLPickUpAttempt(ActionDescription):
    object_designator: Body
    number: int

    @property
    def _action_plan(self):
        return sequential([task_stage(TaskStage.PREPARE_ATTEMPT),
                           RLPickUpAction(self.object_designator)])


@dataclass(eq=False, repr=False)
class RLPickUpAlternatives(UnderspecifiedNode):
    """Use CRAM's query/alternative executor for an externally executed skill.

    The stock node trials the *entire* action in a kinematic world. A physical
    RL grasp cannot be evaluated there. Its applicability is checked from live
    measurements and its Giskard preparation/navigation are probed separately.
    Expanding this node still has no control side effects.
    """

    def advance(self):
        action=self._pull_next_action()
        if action is None:
            return False
        self._attach(action)
        self.current_candidate.notify()
        return True

    def parse(self):
        return PickUpAlternativesExecutable(node=self,context=self.context)


@dataclass
class PickUpAlternativesExecutable(UnderspecifiedExecutable):
    def execute(self):
        try:
            while self.node.advance():
                try:
                    # perform() records the complete attempt's lifecycle, including
                    # preparation, rather than marking it done after its first step.
                    self.node.current_candidate.perform()
                    return self.node.current_candidate.result
                except RECOVERABLE_FAILURES:
                    continue
            raise EmptyUnderspecified()
        finally:
            self.node.stop_grounding()


@dataclass
class AdaptivePickUpRuntime(DemoPickUpRuntime):
    attempted_references: list = field(default_factory=list)
    rejected_stations: list = field(default_factory=list)
    selected_reference: str | None = None
    selection_active: bool = False
    failed_grasp: bool = False
    abort_reason: str | None = None
    attempt_number: int = 0

    def resolve_pickup(self):
        maximum=getattr(self.demo.options,'max_grasp_attempts',3)
        if not 1 <= maximum <= 10:
            raise ValueError('max_grasp_attempts must be between 1 and 10')
        self.attempted_references.clear(); self.rejected_stations.clear()
        self.abort_reason=None; self.failed_grasp=False; self.attempt_number=0
        query=a(RLPickUpAttempt)(object_designator=self.demo.body,
                                number=variable(int, domain=list(range(1,maximum+1))))
        root=execute_single(RLPickUpAlternatives(underspecified_action=query), context=self.demo.context)
        mount_task_motion(self.demo.task_plan,root)
        self.selection_active=True
        try:
            root.perform()
        except EmptyUnderspecified as error:
            raise RLPickUpFailure(self.abort_reason or 'Grasp attempt budget exhausted') from error
        finally:
            self.selection_active=False
            self.demo.record('pickup_resolution_finished', attempts=self.attempt_number,
                references=self.attempted_references, abort_reason=self.abort_reason)

    def stop(self, reason):
        self.abort_reason=str(reason)
        self.demo.record('pickup_recovery_blocked', reason=self.abort_reason)
        raise RLPickUpFailure(self.abort_reason)

    def prepare_attempt(self):
        if self.abort_reason is not None:
            raise RLPickUpFailure(self.abort_reason)
        self.attempt_number+=1
        self.demo.record('pickup_attempt_started', attempt=self.attempt_number,
                         maximum=getattr(self.demo.options, 'max_grasp_attempts', 3))
        description=self.demo.rl.describe_skill()
        if not description['eligible']:
            self.stop(description['rejection'])
        if description['snapshot']['grasp_measurements']['contact']:
            self.stop('The gripper is in contact with the object; automatic preparation is unsafe')
        expected_object = getattr(self.demo, 'skill_object_name', 'bowl')
        if description['skill']['object_name'] != expected_object or description['skill']['arm'] != 'RIGHT':
            self.stop(f'The deployed skill does not match this {expected_object}/right-arm task')
        preferred=self.demo.options.reference
        try:
            choose_reference(description,self.attempted_references,preferred)
        except ValueError as error:
            self.stop(error)
        self.demo.record('pickup_applicability', attempt=self.attempt_number)
        if self.failed_grasp:
            state=self.demo.rl.snapshot()
            reason=recovery_rejection(state,description)
            if reason is not None:
                self.stop(reason)
            # Plan while the bridge still holds the robot. Only a verified
            # recovery request may hand control back to Giskard.
            self.demo.record('pickup_recovery_probe_started')
            if not self.demo.probe_preparation(open_gripper=True):
                self.stop('No collision-free recovery to the starting posture')
            self.demo.rl.recover(state['request_id'])
            self.demo.record('pickup_recovery_started', request_id=state['request_id'])
            try:
                self.demo.open_gripper_verified()
                self.demo.training_posture('recover_rl_posture')
            except Exception as error:
                self.stop(f'Recovery motion failed: {error}')
            self.failed_grasp=False
        else:
            if self.demo.rl.snapshot()['owner'] != 'cram':
                self.stop('The robot controller is not available')
            self.demo.training_posture('prepare_rl_posture')
        description=self.demo.rl.describe_skill()
        if not description['eligible']:
            self.stop(description['rejection'])
        choices=station_candidates(description, self.demo.options.base, self.rejected_stations)
        self.demo.record('station_candidates', candidates=choices, probe_limit=6)
        station=None
        for candidate in choices[:6]:
            self.demo.record('station_probe_started', goal=candidate)
            if self.demo.probe_navigation(candidate):
                station=candidate; break
            self.rejected_stations.append(candidate)
        if station is None:
            self.stop('No navigation candidate passed the collision-aware planning-world check')
        self.demo.record('pickup_station_selected', attempt=self.attempt_number,
            station=station, object=description['snapshot']['object_map_position'],
            checkpoint_sha256=description['skill']['checkpoint_sha256'])
        try:
            self.demo.navigate_to_grasp_region(station)
        except Exception:
            self.rejected_stations.append(station)
            raise
        deadline=time.monotonic()+5.
        while True:
            description=self.demo.rl.describe_skill()
            if not description['eligible']:
                self.stop(description['rejection'])
            if description['admission'] is None:
                break
            if time.monotonic() >= deadline:
                self.rejected_stations.append(station)
                raise RLPickUpFailure(description['admission'])
            time.sleep(.1)
        self.selected_reference=choose_reference(description,self.attempted_references,preferred)
        self.demo.record('pickup_reference_selected', attempt=self.attempt_number,
            reference=self.selected_reference, allowed=description['allowed_references'],
            already_attempted=list(self.attempted_references))

    def pick_up(self,body,arm,reference,timeout):
        if not self.selection_active:
            return super().pick_up(body,arm,reference,timeout)
        reference=self.selected_reference
        if reference is None or reference in self.attempted_references:
            raise RLPickUpFailure('No fresh grounded grasp reference')
        self.attempted_references.append(reference)
        try:
            return super().pick_up(body,arm,reference,timeout)
        except Exception as error:
            self.failed_grasp=True
            self.demo.record('pickup_attempt_failed', attempt=self.attempt_number,
                             reference=reference, reason=str(error))
            raise

    def perform_stage(self,stage):
        if stage == TaskStage.RESOLVE_PICKUP:
            return self.resolve_pickup()
        if stage == TaskStage.PREPARE_ATTEMPT:
            return self.prepare_attempt()
        return super().perform_stage(stage)
