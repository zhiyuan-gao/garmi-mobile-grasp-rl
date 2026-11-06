"""Sequence status must not finish with the bowl or mutate the CRAM plan."""
from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import sequential
from semantic_digital_twin.world import World
from garmi_cram_rl.pick_up_action import task_stage, TaskStage
from household_plan import HouseholdDecisionBridge


def test_observed_sequence_keeps_actual_plan_and_waits_for_final_verification(tmp_path):
    root = sequential([task_stage(TaskStage.TRANSPORT)], context=Context(world=World(), robot=None))
    before = list(root.plan.nodes)
    bridge = HouseholdDecisionBridge(decision_archive=tmp_path/'plan.json')
    bridge.begin_plan(root.plan)
    bridge.observe_decision(dict(event='household_stage', stage='mug_rl_approach'))
    nodes = {n.id:n for n in bridge.plan_state.nodes}
    assert nodes['household_sequence'].status == 'RUNNING'
    assert nodes['household_mug_rl_approach'].status == 'RUNNING'
    assert nodes['household_close_drawer_left'].status == 'CREATED'
    bridge.observe_decision(dict(event='household_stage', stage='close_drawer_left'))
    bridge.observe_decision(dict(event='household_stage', stage='completed'))
    assert bridge.sequence_status == 'RUNNING'
    bridge.observe_decision(dict(event='household_completed'))
    assert bridge.sequence_status == 'SUCCEEDED'
    assert list(root.plan.nodes) == before
    bridge.snapshot_plan()
    assert sum(n.id == 'household_sequence' for n in bridge.plan_state.nodes) == 1


def test_failed_sequence_does_not_mark_active_stage_succeeded():
    root = sequential([task_stage(TaskStage.TRANSPORT)], context=Context(world=World(), robot=None))
    bridge = HouseholdDecisionBridge()
    bridge.begin_plan(root.plan)
    bridge.observe_decision(dict(event='household_stage', stage='mug_rl_approach'))
    bridge.observe_decision(dict(event='failed', error='physical grasp failed'))
    assert bridge.sequence_status == 'FAILED'
    assert bridge.sequence_stages['mug_rl_approach'] == 'FAILED'
