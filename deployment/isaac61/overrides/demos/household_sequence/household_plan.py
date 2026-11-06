"""Show measured sequence progress alongside the actual CRAM action tree."""
from dataclasses import dataclass, field, replace
import hashlib
import json
from garmi_cram_rl.decision_view import DecisionBridge
from cramera.live.bridge import PlanNodeEntry, PlanSnapshot
from cramera.knowledge.enums import PlanNodeGroup

STAGES = {
    'return_to_counter_mug': 'Return to counter',
    'mug_rl_approach': 'Mug · RL approach and fixed close/lift',
    'carry_mug_to_drawer': 'Carry mug to drawer',
    'open_drawer_left': 'Open drawer · left arm IK',
    'place_mug_inside': 'Place mug inside · right arm IK',
    'close_drawer_left': 'Close drawer · left arm IK',
}

@dataclass
class HouseholdDecisionBridge(DecisionBridge):
    sequence_status: str = 'RUNNING'
    sequence_stages: dict = field(default_factory=dict)

    def observe_decision(self, event):
        with self.decision_lock:
            if event['event'] in ('household_stage', 'household_completed', 'failed'):
                stage = event.get('stage')
                # Stage transitions come from the executing simulation, not a timer.
                for key in self.sequence_stages:
                    if self.sequence_stages[key] == 'RUNNING':
                        self.sequence_stages[key] = 'FAILED' if event['event'] == 'failed' else 'SUCCEEDED'
                if stage in STAGES:
                    self.sequence_stages[stage] = 'RUNNING'
                if event['event'] == 'household_completed':
                    self.sequence_status = 'SUCCEEDED'
                elif event['event'] == 'failed':
                    self.sequence_status = 'FAILED'
            super().observe_decision(event)

    def snapshot_plan(self):
        with self.decision_lock:
            super().snapshot_plan()
            if self._plan is None:
                return
            def node(key, label, status, parent, kind):
                return PlanNodeEntry(id=key, parent=parent, label=label, status=status,
                    kind=kind, group=PlanNodeGroup.of_plan_node_kind(kind), derived=False)
            entries = [node('household_sequence', 'Bowl to table → mug into drawer',
                self.sequence_status, None, 'SequentialNode')]
            entries += [replace(n, parent='household_sequence') if not n.parent else n
                        for n in self.plan_state.nodes]
            # Monitor rows describe the existing controller, not new CRAM actions.
            entries += [node('household_' + key, label,
                self.sequence_stages.get(key, 'CREATED'), 'household_sequence', 'MonitorNode')
                for key, label in STAGES.items()]
            signature = hashlib.sha256(json.dumps([(n.id,n.parent,n.label) for n in entries]).encode()).hexdigest()
            with self._lock:
                self.plan_state = PlanSnapshot(signature=signature, nodes=entries)
            if self.decision_archive:
                self.decision_archive.write_text(json.dumps(self.plan_state.to_payload()) + '\n')
