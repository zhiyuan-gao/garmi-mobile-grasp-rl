"""The real CRAM plan and NeemHub, with perception provenance in its decision rows."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass, field
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'integration'), str(ROOT/'scripts'), str(ROOT/'demos/robokudo_mug'), str(ROOT/'cram-vrb-lab')]
from web_demo import WebDemo
from cram_demo import GraspDemo
from mug_task import MugDemo
from garmi_cram_rl.decision_view import DecisionBridge, DecisionTrace
from mobile_grasp import cancel_motion

@dataclass
class PerceptionTrace(DecisionTrace):
    def observe(self, event, parent):
        super().observe(event,parent)
        obj=os.environ.get('GARMI_PERCEPTION_OBJECT','mug')
        if event['event']=='pickup_applicability':
            self.rows[(self.attempt,'applicability')]['label']=f'Skill applicable · {obj} / right arm'
        if event['event']=='pickup_attempt_started':
            estimate=json.loads(Path(os.environ['GARMI_PERCEPTION_RESULT']).read_text())
            x,y,z=estimate['position_map']
            for i,label in enumerate([
                'RoboKudo · head RGB-D snapshot captured',
                'RoboKudo · support plane and object point cloud extracted',
                f'Model fit · {obj} at ({x:.3f}, {y:.3f}, {z:.3f}) m · orientation fixed',
                'CRAM station and RL target use this perceived position']):
                self.rows[(self.attempt,f'perception-{i}')]=dict(id=f'decision_perception_{i}',parent=parent,label=label,status='SUCCEEDED')

@dataclass
class PerceptionBridge(DecisionBridge):
    decision_trace: DecisionTrace = field(default_factory=PerceptionTrace)

def attach_viewer(demo):
    from cramera.live.visualization import LiveVisualization
    from cramera.scene_presentation import ScenePresentation
    port=int(os.environ.get('GARMI_LIVE_PORT','8767'))
    demo.viewer=LiveVisualization(demo.world,port=port,bridge=demo.bridge_type(
        decision_archive=demo.output/'decision-plan.json'))
    demo.viewer.bridge.presentation=ScenePresentation(
        camera={'position':[1.2,6.1,2.1],'target':[.15,7.05,1.05]})
    demo.viewer.start()
    demo.record('viewer_ready',port=port)

class LiveMug(MugDemo,WebDemo):
    bridge_type=PerceptionBridge

    def setup_object(self):
        # MugDemo replaces object creation, so attach the native viewer here.
        MugDemo.setup_object(self)
        attach_viewer(self)

class LiveBowl(WebDemo):
    bridge_type=PerceptionBridge

    def setup_object(self):
        GraspDemo.setup_object(self)
        attach_viewer(self)

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--object',choices=['bowl','mug'],required=True)
a=p.parse_args()
o=argparse.Namespace(reference='mug-v3-ref025' if a.object=='mug' else None,base=None,
    place_base=[.9,5.85,-.85],target=[1.6,5.32,.8],stage='rl-transport',carry='physical',
    transport=False,resume_place=False,retreat=None,grasp_camera_offset=[-.55,.28,.85],max_grasp_attempts=1)
demo=(LiveMug if a.object=='mug' else LiveBowl)(o,output=a.out)
try:
    demo.run()
    state=demo.rl.snapshot()
    (a.out/'result.json').write_text(json.dumps(dict(status='completed',result=state['result'],snapshot=state['snapshot']),indent=2))
    demo.record('completed',stage='robokudo_pick_up')
    while True: time.sleep(1)  # Keep the actual final world and plan available.
except BaseException as error:
    if hasattr(demo,'context'):cancel_motion(demo.context)
    demo.record('failed',error=repr(error))
    raise
finally:
    if hasattr(demo,'executor'):demo.close()
