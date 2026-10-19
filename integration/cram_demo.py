"""Original CRAM navigation -> RL physical grasp -> CRAM control handback."""
import argparse
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'cram-vrb-lab')]

# Bind the checked-out robot package before the legacy demo adds its image path.
import cram_vrb_lab

from mobile_grasp import (
    MobileGrasp, Bowl, BodySpecification, Connection6DoFSpecification,
    HomogeneousTransformationMatrix, NavigateAction, cancel_motion,
    Arms, PlaceAction,
)
from garmi_cram_rl.client import GraspClient
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion


class GraspDemo(MobileGrasp):
    def training_posture(self, label):
        self.run_motion(label, MoveJointsMotion(
            names=[f'right_fr3_joint{i}' for i in range(1, 8)],
            positions=[0., -math.pi / 4, 0., -3 * math.pi / 4, 0., math.pi / 2, math.pi / 4]))

    def connect(self):
        super().connect()
        self.rl = GraspClient(self.node)
        self.rl.snapshot()

    def setup_object(self):
        # The persistent simulator owns the physical bowl. Only register its
        # measured pose and mesh in CRAM's planning world; never spawn a duplicate.
        state = self.rl.snapshot()['snapshot']
        name = 'graspgen_bowl.stl'
        initial = HomogeneousTransformationMatrix.from_xyz_rpy(*state['object_map_position'])
        if not self.world.is_kinematic_structure_entity_in_world_by_name(name):
            Bowl.get_annotation_specification(name,
                BodySpecification.mesh(name, str(self.mesh_path), parent_T_self=initial),
                parent_connection_specification=Connection6DoFSpecification()).spawn(self.world)
        self.body = self.world.get_body_by_name(name)
        self.annotation = next(a for a in self.world.get_semantic_annotations_by_type(Bowl) if a.root is self.body)
        # Register the existing simulator name with the legacy pose decoder.
        # place()/apply() would spawn or teleport the object, so neither is called.
        if name not in self.sync._synced:
            self.sync._synced.append(name)
        held = self.body.parent_kinematic_structure_entity is not self.world.root
        if held and not self.options.resume_place:
            raise RuntimeError('CRAM still holds a previous bowl; reconcile the task before retrying')
        if self.options.resume_place and not held:
            raise RuntimeError('Placement continuation requires an existing semantic grasp')
        deadline = time.monotonic() + 5.
        while name not in self.sync.tracked_poses():
            if time.monotonic() > deadline:
                raise TimeoutError('Simulator did not report the existing bowl pose')
            time.sleep(.05)
        self.sync.pull(self.world)
        self.record('settled', object=self.object_pose(), output_dir=str(self.output), backend='rl_training_assets')

    def pick(self):
        cancel_motion(self.context)
        bowl = self.rl.snapshot()['snapshot']['object_map_position']
        offset = getattr(self.options, 'grasp_camera_offset', [-.55, .28, .85])
        self.record('camera', eye=[p + d for p, d in zip(bowl, offset)],
                    target=[bowl[0], bowl[1], bowl[2] + .12])
        time.sleep(.5)
        result = self.rl.grasp(self.options.reference)
        self.sync.pull(self.world)
        # Semantic held relation only: simulator always keeps physical contacts.
        self.world.move_branch(self.body, self.tool)
        self.record('rl_grasp_result', **result)
        self.record('cram_control_returned', object=self.object_pose(), automatic_attachment=False)
        self.verify_hold()

    def verify_hold(self):
        time.sleep(2.)
        state = self.rl.snapshot()
        measured = state['snapshot']['grasp_measurements']
        passed = bool(state['owner'] == 'cram' and measured and measured['contact']
                      and measured['height'] >= .08 and measured['slip'] < .02
                      and not measured['supported'] and not measured['collision'])
        self.record('handover_hold_check', passed=passed, measurements=measured)
        if not passed:
            raise RuntimeError('Physical grasp was not preserved after CRAM handback')

    def navigate_to_grasp_region(self):
        x, y, yaw = self.options.base
        target = HomogeneousTransformationMatrix.from_xyz_rpy(x, y, 0., yaw=yaw).to_pose()
        target.reference_frame = self.world.root
        self.run_motion('navigate_to_grasp_region', NavigateAction(target_location=target, keep_joint_states=True))
        cancel_motion(self.context)
        time.sleep(.5)
        measured = self.rl.snapshot()['snapshot']
        self.record('navigation_arrival', requested=self.options.base,
                    actual=measured['base_map_position'], yaw=measured['base_map_yaw'],
                    point_error_m=math.dist(measured['base_map_position'][:2], self.options.base[:2]),
                    acceptance='RL admission checks the actual trained region, posture and rest state')

    def run(self):
        self.connect()
        self.setup_object()
        self.record('camera', eye=[-1.2, 4.4, 2.7], target=[.7, 6.3, 1.])
        if self.options.transport and not self.rl.snapshot()['snapshot']['full_scene']:
            raise RuntimeError('Dining-table transport requires --full-scene in the simulator')
        if self.options.resume_place:
            self.verify_hold()
            self.place()
            return
        self.training_posture('prepare_rl_posture')
        self.navigate_to_grasp_region()
        self.pick()
        if self.options.transport:
            self.transport()
            return
        if self.options.retreat:
            self.navigate(self.options.retreat)
            self.check_carry()
            self.verify_hold()

    def transport(self):
        # First clear the counter with the physically held bowl and the arm fixed.
        self.record('camera', eye=[-1.2, 4.4, 2.7], target=[.7, 6.3, 1.])
        self.navigate([.25, 5.75, math.pi / 2])
        self.check_carry()
        self.record('carry_posture', mode='keep_measured_joint_states',
                    reason='The generic parked pose brings this held bowl into the fixed left hand')
        self.navigate(self.options.place_base)
        self.check_carry()
        self.place()

    def place(self):
        self.record('camera', eye=[-.8, 3.6, 2.6], target=[1.2, 5.35, .95])
        yaw = self.rl.snapshot()['snapshot']['object_map_euler'][2]
        target = HomogeneousTransformationMatrix.from_xyz_rpy(*self.options.target, yaw=yaw).to_pose()
        target.reference_frame = self.world.root
        self.record('placement_target', position=self.options.target, yaw=yaw,
                    orientation='upright, preserve the carried bowl heading')
        self.check_place(target)
        self.run_motion('place', PlaceAction(object_designator=self.body,
            target_location=target, arm=Arms.RIGHT, approach_clearance=.08,
            retreat_distance=.12), timeout=120)
        time.sleep(2.)
        # Verify physical release/support, rather than just the symbolic plan result.
        samples = []
        for _ in range(21):
            samples.append(self.rl.snapshot()['snapshot'])
            time.sleep(.1)
        released = self.body.parent_kinematic_structure_entity is self.world.root
        errors = [math.dist(s['object_map_position'][:2], self.options.target[:2]) for s in samples]
        supported = all(s['grasp_measurements']['supported'] for s in samples)
        unheld = all(not s['grasp_measurements']['contact'] for s in samples)
        stationary = all(math.sqrt(sum(v*v for v in s['object_velocity'][:3])) < .02 for s in samples)
        height_ok = all(.77 < s['object_map_position'][2] < .83 for s in samples)
        passed = bool(released and max(errors) < .05 and supported and unheld and stationary and height_ok)
        self.record('transport_result', passed=passed, carry_mode='physical', automatic_attachment=False,
                    object_position=samples[-1]['object_map_position'], horizontal_error_m=max(errors),
                    released=released, supported=supported, no_finger_contact=unheld,
                    stationary=stationary, height_ok=height_ok, target=self.options.target,
                    verification_seconds=2.)
        if not passed:
            raise RuntimeError('Bowl did not settle released and physically supported on the dining table')
        self.training_posture('finish_park')
        time.sleep(2.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', help='One of the 40 trained ids; otherwise first allowed mask entry')
    parser.add_argument('--grasp-camera-offset', type=float, nargs=3, default=[-.55, .28, .85],
                        help='Recording camera offset from the measured bowl in map coordinates')
    parser.add_argument('--base', type=float, nargs=3, default=[.25, 6.35, math.pi / 2])
    parser.add_argument('--retreat', type=float, nargs=3, help='Optional CRAM base goal after physical grasp')
    parser.add_argument('--transport', action='store_true', help='Carry and place at the dining table')
    parser.add_argument('--drawer', action='store_true', help='Left-hand open/place/close upper drawer')
    parser.add_argument('--resume-drawer', choices=['open', 'place', 'close'], help='Debug continuation in the same physical scene')
    parser.add_argument('--resume-place', action='store_true', help='Continue a held bowl after a placement planning failure')
    parser.add_argument('--place-base', type=float, nargs=3, default=[.90, 5.85, -.85])
    parser.add_argument('--target', type=float, nargs=3, default=[1.6, 5.32, .80])
    parser.add_argument('--output-dir', type=Path, required=True)
    options = parser.parse_args()
    options.stage, options.carry = ('rl-drawer' if options.drawer else 'rl-transport' if options.transport else 'rl-pick'), 'physical'
    if options.resume_drawer in ('open', 'place'):
        options.resume_place = True
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    if options.drawer:
        from drawer_demo import DrawerDemo
        demo = DrawerDemo(options, output=options.output_dir)
    else:
        demo = GraspDemo(options, output=options.output_dir)
    try:
        demo.run()
        demo.record('completed', stage=options.stage)
    except BaseException as error:
        if hasattr(demo, 'context'):
            cancel_motion(demo.context)
        demo.record('failed', error=repr(error))
        raise
    finally:
        if hasattr(demo, 'executor'):
            demo.close()


if __name__ == '__main__':
    main()
