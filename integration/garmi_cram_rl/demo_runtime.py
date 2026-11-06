"""Adapt the validated counter demo to CRAM's learned pick-up action."""
from dataclasses import dataclass

from coraplex.datastructures.enums import Arms
from garmi_cram_rl.pick_up_action import RLPickUpFailure, TaskStage


@dataclass
class DemoPickUpRuntime:
    demo: object

    @property
    def world(self):
        return self.demo.world

    def pick_up(self, body, arm, reference, timeout):
        # This bridge owns one verified policy/asset binding. A second object
        # must have its own binding, not silently use the bowl policy.
        if body is not self.demo.body or arm != Arms.RIGHT:
            raise RLPickUpFailure('No deployed policy is bound to this object and arm')
        if body.parent_kinematic_structure_entity is not self.world.root:
            raise RLPickUpFailure('The target is already held or has an unsupported parent')
        if self.demo.rl.snapshot()['owner'] != 'cram':
            raise RLPickUpFailure('The grasp controller is not ready for a new request')
        self.demo.prepare_grasp_view()
        self.demo.record('rl_grasp_requested', reference=reference)
        result = self.demo.rl.grasp(reference, timeout=timeout)
        if not result['result']['success']:
            raise RLPickUpFailure(result['result']['reason'])
        # Verify the actual hold *before* asserting a symbolic grasp. Do not
        # attach anything in PhysX or substitute the reference for the measured pose.
        measurements = self.demo.verify_hold()
        self.demo.sync.pull(self.world)
        self.world.move_branch(body, self.demo.tool)
        self.demo.record('rl_grasp_result', **result)
        self.demo.record('cram_control_returned', object=self.demo.object_pose(), automatic_attachment=False)
        return dict(success=True, object=str(body.name), arm=arm.name,
                    reference=result['snapshot']['reference'], measurements=measurements,
                    request_id=result['request_id'])

    def perform_stage(self, stage):
        if stage == TaskStage.PREPARE:
            self.demo.training_posture('prepare_rl_posture')
        elif stage == TaskStage.NAVIGATE:
            self.demo.navigate_to_grasp_region()
        elif stage == TaskStage.TRANSPORT:
            self.demo.transport()
        elif stage == TaskStage.RETREAT:
            self.demo.navigate(self.demo.options.retreat)
            self.demo.check_carry()
            self.demo.verify_hold()
        else:
            raise ValueError(f'Unknown task stage: {stage}')
