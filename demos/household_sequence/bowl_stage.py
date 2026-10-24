"""The existing bowl routine, with the already-present counter mug in its planning world."""
import argparse
import math
from pathlib import Path
from scipy.spatial.transform import Rotation
from cram_demo import GraspDemo, cancel_motion
from mobile_grasp import BodySpecification, Connection6DoFSpecification, HomogeneousTransformationMatrix
from semantic_digital_twin.semantic_annotations.semantic_annotations import Mug

class BowlStage(GraspDemo):
    def setup_object(self):
        super().setup_object()
        state=self.rl.snapshot()['snapshot']
        q=state['mug_map_quaternion_wxyz']
        rpy=Rotation.from_quat([*q[1:],q[0]]).as_euler('xyz')
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(*state['mug_map_position'],*rpy)
        name='household_counter_mug'
        mesh='/mug/assets/ycb/025_mug/surface-v3/repaired.obj'
        Mug.get_annotation_specification(name,BodySpecification.mesh(name,mesh,parent_T_self=pose),
            parent_connection_specification=Connection6DoFSpecification()).spawn(self.world)
        self.record('mug_planning_obstacle',position=state['mug_map_position'],mesh=mesh)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    options=parser.parse_args()
    if options.output_dir.exists():raise FileExistsError(options.output_dir)
    options.reference='bowl-r020';options.base=[.25,6.35,math.pi/2]
    options.place_base=[.90,5.85,-.85];options.target=[1.6,5.32,.80]
    options.stage='rl-transport';options.carry='physical';options.transport=True
    options.resume_place=False;options.retreat=None
    demo=BowlStage(options,output=options.output_dir)
    try:
        demo.run();demo.record('completed',stage=options.stage)
    except BaseException as error:
        if hasattr(demo,'context'):cancel_motion(demo.context)
        demo.record('failed',error=repr(error));raise
    finally:
        if hasattr(demo,'executor'):demo.close()

if __name__=='__main__':main()
