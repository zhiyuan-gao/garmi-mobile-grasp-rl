"""Keep the CRAM world and NeemHub alive through the complete physical task."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'integration'), str(ROOT/'scripts')]
from web_demo import WebDemo
from mobile_grasp import BodySpecification, Connection6DoFSpecification, HomogeneousTransformationMatrix, cancel_motion
from semantic_digital_twin.semantic_annotations.semantic_annotations import Mug

class HouseholdLiveTask(WebDemo):
    def setup_object(self):
        super().setup_object()
        state = self.rl.snapshot()['snapshot']
        name = 'household_counter_mug'
        mesh = str(ROOT/'household/mug/assets/ycb/025_mug/surface-v3/repaired.obj')
        pose = self.measured_pose(state['mug_map_position'], state['mug_map_quaternion_wxyz'], self.world.root)
        Mug.get_annotation_specification(name, BodySpecification.mesh(name, mesh, parent_T_self=pose),
            parent_connection_specification=Connection6DoFSpecification()).spawn(self.world)
        self.mug_body = self.world.get_body_by_name(name)
        # The twin already contains the authored drawer and its slide joint.
        self.drawer_body = self.world.get_body_by_name('drawer_1')
        self.drawer_initial = np.asarray(self.drawer_body.global_pose.to_np()).copy()
        self.last_sync_record = -1.
        self.sync_subscription = self.node.create_subscription(
            __import__('std_msgs.msg', fromlist=['String']).String,
            '/fame/rl_grasp/state', self.on_measured_state, 1)
        self.record('household_live_ready', mug=name, drawer=str(self.drawer_body.name))

    @staticmethod
    def measured_pose(position, q, frame):
        data = np.eye(4); data[:3,3] = position
        data[:3,:3] = Rotation.from_quat([*q[1:],q[0]]).as_matrix()
        return HomogeneousTransformationMatrix(data=data,reference_frame=frame)

    def on_measured_state(self, message):
        try:
            self.apply_measured_state(message)
        except Exception as error:
            self.record('live_sync_failed',error=repr(error))
            raise

    def apply_measured_state(self, message):
        state = json.loads(message.data)['snapshot']
        if not hasattr(self, 'mug_body'):
            return
        self.mug_body.parent_connection.origin = self.measured_pose(
            state['mug_map_position'], state['mug_map_quaternion_wxyz'],self.world.root)
        # Move only the twin. The simulated drawer remains contact driven.
        self.drawer_body.parent_connection.position = state['drawer_opening_m']
        self.sync.pull(self.world)
        self.world.notify_state_change()
        now = state['simulation_seconds']
        if now - self.last_sync_record >= .5:
            with (self.output/'live-sync.jsonl').open('a') as trace:
                trace.write(json.dumps({'simulation_seconds':now,'stage':state['sequence_stage'],
                    'mug_map_position':state['mug_map_position'], 'drawer_opening_m':state['drawer_opening_m']})+'\n')
            self.last_sync_record = now


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    o=p.parse_args()
    if o.output_dir.exists():raise FileExistsError(o.output_dir)
    o.reference='bowl-r020';o.base=[.25,6.35,math.pi/2];o.place_base=[.9,5.85,-.85];o.target=[1.6,5.32,.8]
    o.grasp_camera_offset=[-.55,.28,.85];o.stage='rl-transport';o.carry='physical';o.transport=True
    o.resume_place=False;o.retreat=None
    demo=HouseholdLiveTask(o,output=o.output_dir)
    try:
        demo.run();demo.record('completed',stage='bowl_counter_to_table')
        (o.output_dir.parent/'continue-mug.json').write_text(json.dumps({'bowl_delivery_verified':True}))
        last=None
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            sequence=o.output_dir.parent/'sim/sequence.json'
            report=json.loads(sequence.read_text())
            stage=demo.rl.snapshot()['snapshot']['sequence_stage']
            if stage != last:
                demo.record('household_stage',stage=stage);last=stage
            if report['status']=='failed':raise RuntimeError(report.get('error','Household task failed'))
            if report['status']=='completed':
                demo.record('household_completed',verification=report)
                # Retain both live panes at the measured final state until Stop.
                while True:time.sleep(1)
            time.sleep(.1)
        raise TimeoutError('Household task timeout')
    except BaseException as error:
        if hasattr(demo,'context'):cancel_motion(demo.context)
        demo.record('failed',error=repr(error));raise
    finally:
        if hasattr(demo,'executor'):demo.close()

if __name__=='__main__':main()
