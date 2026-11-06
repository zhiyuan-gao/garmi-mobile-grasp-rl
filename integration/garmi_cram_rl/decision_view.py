"""Presentation-only annotations for the existing NeemHub plan tree.

Execution nodes keep their identity and status. Event rows describe observations,
not additional CRAM actions; they use the viewer's existing monitor-detail rows.
"""
from dataclasses import dataclass, field
import hashlib
import json
import threading

from cramera.live.bridge import Bridge, PlanNodeEntry, PlanSnapshot
from cramera.knowledge.enums import PlanNodeGroup
from coraplex.plans.plan_node import ActionNode
from garmi_cram_rl.pick_up_action import SkillNode, TaskStage


def plain(value):
    # The pinned Steps renderer inserts labels as HTML. Keep diagnostics as text.
    return str(value).replace('&', '＆').replace('<', '‹').replace('>', '›')


def xy(value):
    return f'({value[0]:.2f}, {value[1]:.2f}) m'


@dataclass
class DecisionTrace:
    rows: dict = field(default_factory=dict)
    attempt: int = 0
    maximum: int = 3
    reference: str | None = None
    probe: int = 0

    def observe(self, event, parent):
        name = event['event']
        if name == 'pickup_attempt_started':
            self.attempt = event['attempt']
            self.maximum = event['maximum']
            self.reference = None
            self.probe = 0

        def row(key, label, status='SUCCEEDED'):
            key = (self.attempt, key)
            previous = self.rows.get(key)
            self.rows[key] = dict(id=previous['id'] if previous else f'decision_{len(self.rows)}',
                parent=previous['parent'] if previous else parent,
                label=plain(label), status=status)

        if name == 'pickup_attempt_started':
            row('applicability', 'Check skill applicability', 'RUNNING')
        elif name == 'pickup_applicability':
            row('applicability', 'Skill applicable · bowl / right arm')
        elif name == 'station_candidates':
            row('candidates', f"Select station · {len(event['candidates'])} candidates; at most {event['probe_limit']} checks")
            row('ordering', 'Order: lateral alignment with object, then navigation distance')
        elif name == 'station_probe_started':
            self.probe += 1
            row(f'route-{self.probe}', f"Check route to {xy(event['goal'])} · planning-world copy", 'RUNNING')
        elif name == 'station_probe':
            # Also accept historical logs, which have only the completed probe.
            self.probe = max(1, self.probe)
            passed = event['passed']
            label = f"Route to {xy(event['goal'])} · {'passed' if passed else 'rejected'}"
            if 'solver_steps' in event:
                label += f" · {event['solver_steps']} solver steps / {event['path_length_m']:.2f} m"
            row(f'route-{self.probe}', label, 'SUCCEEDED' if passed else 'FAILED')
            if not passed:
                row(f'route-reason-{self.probe}', event.get('error', 'Final pose outside arrival tolerance'), 'FAILED')
        elif name == 'pickup_station_selected':
            row('station', f"Selected station · {xy(event['station'])}")
        elif name == 'motion_start' and event['label'] == 'navigate_to_grasp_region':
            row('navigation', 'Navigate to selected station', 'RUNNING')
        elif name == 'navigation_arrival':
            row('navigation', f"Navigation arrived · error {event['point_error_m']*100:.1f} cm")
        elif name == 'pickup_reference_selected':
            self.reference = event['reference']
            row('reference', f"Select grasp · {self.reference} · {len(event['allowed'])} allowed")
            row('excluded', 'Previously tried: ' + (', '.join(event['already_attempted']) or 'none'))
        elif name == 'rl_grasp_requested':
            self.reference = event.get('reference') or self.reference
            row('grasp', f"RL approach → fixed close/lift · {self.reference or 'automatic reference'}", 'RUNNING')
        elif name == 'handover_hold_check':
            passed = event['passed']
            row('grasp', f"Grasp {'verified' if passed else 'not held'} · lift {event['measurements']['height']*100:.1f} cm",
                'SUCCEEDED' if passed else 'FAILED')
        elif name == 'pickup_attempt_failed':
            row('grasp', f"Grasp failed · {event['reason']}", 'FAILED')
        elif name == 'pickup_recovery_probe_started':
            row('recovery', 'Check recovery · planning-world copy', 'RUNNING')
        elif name == 'recovery_probe':
            row('recovery', 'Recovery path ' + ('passed' if event['passed'] else 'rejected'),
                'SUCCEEDED' if event['passed'] else 'FAILED')
        elif name == 'pickup_recovery_started':
            row('recover-motion', 'Open fingers and restore starting posture', 'RUNNING')
        elif name == 'motion_done' and event['label'] == 'recover_rl_posture':
            row('recover-motion', 'Posture restored · choose an untried grasp')
        elif name in ('pickup_recovery_blocked', 'failed'):
            for item in self.rows.values():
                if item['status'] == 'RUNNING':
                    item['status'] = 'FAILED'
            row('stop', 'Stopped · ' + str(event.get('reason', event.get('error'))), 'FAILED')
        elif name == 'transport_result':
            row('placement', f"Placement {'verified' if event['passed'] else 'failed'} · error {event['horizontal_error_m']*1000:.1f} mm",
                'SUCCEEDED' if event['passed'] else 'FAILED')


STAGE_LABELS = {
    TaskStage.RESOLVE_PICKUP: 'Choose and execute RL pick-up',
    TaskStage.PREPARE_ATTEMPT: 'Check station, navigate and select grasp',
    TaskStage.PREPARE: 'Prepare right arm', TaskStage.NAVIGATE: 'Navigate to counter',
    TaskStage.TRANSPORT: 'Carry, place and verify', TaskStage.RETREAT: 'Retreat and check hold',
}


@dataclass
class DecisionBridge(Bridge):
    decision_trace: DecisionTrace = field(default_factory=DecisionTrace)
    decision_lock: object = field(default_factory=threading.RLock, repr=False)
    decision_archive: object = None

    def observe_decision(self, event):
        if self._plan is None:
            return
        with self.decision_lock:
            running = [n for n in self._plan.nodes if isinstance(n, SkillNode)
                       and n.status.name == 'RUNNING']
            active = max(running, key=lambda n: len(n.path), default=self._plan.root)
            if isinstance(active, SkillNode) and active.action is not None:
                # CRAM orders path from the immediate parent towards the root.
                active = next((n for n in active.path if isinstance(n, ActionNode)), active.parent)
            self.decision_trace.observe(event, f'plan_node_{id(active)}')
            self.snapshot_plan()

    def snapshot_plan(self):
        with self.decision_lock:
            super().snapshot_plan()
            if self._plan is None:
                return
            by_id = {f'plan_node_{id(n)}': n for n in self._plan.nodes}
            # Mapped actions can execute a controller directly, leaving their
            # expanded primitive templates unexecuted. Do not show those as
            # pending robot motions after the mapped action has completed.
            entries = [n for n in self.plan_state.nodes if not (
                n.kind in ('MotionNode', 'ReAttachNode') and n.status in ('CREATED', 'NOT_STARTED'))]
            for entry in entries:
                node = by_id.get(entry.id)
                if isinstance(node, SkillNode):
                    if node.action is None:
                        entry.kind = 'ActionNode'  # Existing viewer's step-row presentation.
                        entry.group = PlanNodeGroup.of_plan_node_kind(entry.kind)
                        entry.label = STAGE_LABELS[node.stage]
                    else:
                        entry.kind = 'SequentialNode'  # Its actual action is the visible parent.
                elif type(node).__name__ == 'RLPickUpAlternatives':
                    entry.kind = 'UnderspecifiedNode'
                action = getattr(node, 'action', None)
                if type(action).__name__ == 'RLPickUpAttempt':
                    entry.label = f'Attempt {action.number}/{self.decision_trace.maximum}'
                elif type(action).__name__ == 'RLPickUpAction':
                    number = next((n.action.number for n in node.path
                                   if type(getattr(n, 'action', None)).__name__ == 'RLPickUpAttempt'), None)
                    refs = [r['label'] for (attempt, key), r in self.decision_trace.rows.items()
                            if key == 'reference' and attempt == number]
                    reference = refs[-1].split(' · ')[1] if refs else (
                        self.decision_trace.reference if number is None else action.reference)
                    entry.label = 'RLPickUpAction' + (' · ' + plain(reference) if reference else '')
            for item in self.decision_trace.rows.values():
                if item['parent'] in by_id:
                    entries.append(PlanNodeEntry(**item, kind='MonitorNode',
                        group=PlanNodeGroup.of_plan_node_kind('MonitorNode'), derived=True))
            # The existing graph refreshes labels when its signature changes.
            signature = hashlib.sha256(json.dumps([(n.id, n.parent, n.label) for n in entries]).encode()).hexdigest()
            with self._lock:
                self.plan_state = PlanSnapshot(signature=signature, nodes=entries)
            if self.decision_archive is not None:
                payload = self.plan_state.to_payload()
                serialized = json.dumps(payload)
                if serialized != getattr(self, '_last_decision_payload', None):
                    self.decision_archive.parent.mkdir(parents=True, exist_ok=True)
                    self.decision_archive.write_text(serialized + '\n')
                    self._last_decision_payload = serialized
