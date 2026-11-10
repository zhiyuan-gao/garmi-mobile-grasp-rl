"""The existing household plan, annotated with actual perception observations."""
from dataclasses import dataclass, field
import hashlib
import json
from household_plan import HouseholdDecisionBridge
from cramera.live.bridge import PlanNodeEntry, PlanSnapshot
from cramera.knowledge.enums import PlanNodeGroup

@dataclass
class PerceptionHouseholdBridge(HouseholdDecisionBridge):
    perceptions: dict = field(default_factory=dict)

    def observe_decision(self, event):
        with self.decision_lock:
            if event['event'] == 'household_perception':
                self.perceptions[event['observation']] = event
            super().observe_decision(event)

    def snapshot_plan(self):
        with self.decision_lock:
            super().snapshot_plan()
            if self._plan is None:
                return
            nodes = list(self.plan_state.nodes)
            for key,e in self.perceptions.items():
                xyz=', '.join(f'{v:.3f}' for v in e['position_map'])
                nodes.append(PlanNodeEntry(id='household_perception_'+key, parent='household_sequence',
                    label=f"RoboKudo · {key} · ({xyz}) m · " + ('initial CRAM world pose' if key=='mug-initial' else 'position used for grasp'),
                    status='SUCCEEDED', kind='MonitorNode', group=PlanNodeGroup.of_plan_node_kind('MonitorNode'),derived=True))
            signature=hashlib.sha256(json.dumps([(n.id,n.parent,n.label) for n in nodes]).encode()).hexdigest()
            with self._lock:
                self.plan_state=PlanSnapshot(signature=signature,nodes=nodes)
            if self.decision_archive:
                self.decision_archive.write_text(json.dumps(self.plan_state.to_payload())+'\n')
