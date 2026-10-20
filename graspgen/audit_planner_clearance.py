"""Measure nonadjacent open-hand self-clearance at physically tested endpoint seeds."""
import xml.etree.ElementTree as ET
import numpy as np,trimesh,fcl
from garmi_rl.collision_pairs import adjacent_body_pairs
from rl_geometry import RobotCollision
from prepare_wrist_v6 import OUT
from prepare_rl_validation import save
from preview_strict_successes import pose
from select_references import ROOT,sha
from try_handle_roll import read


def main():
    rows=[]
    for name,label in [('pitcher','planner-v4-near'),('mug','planner-v4-handles-near')]:
        folder=OUT/name/label;obj=read(folder/'bank.json')['objects'][0]
        refs={r['id']:r for r in obj['references']};checkers={}
        for c in read(folder/'conditions.json'):
            width=refs[c['reference']]['opening_m']
            if width not in checkers:checkers[width]=RobotCollision(trimesh.creation.box([.001]*3),opening_m=width)
            checker=checkers[width]
            checker.check(pose(c['base_pose'][:3],c['base_pose'][3:]),np.array(c['arm_position']),pose(c['object_state'][:3],c['object_state'][3:7]))
            ignored=adjacent_body_pairs(ET.parse(ROOT/'rl/assets/garmi_training.urdf').getroot(),checker.meshes)
            names=list(checker.meshes);distances=[]
            for i,a in enumerate(names):
                for b in names[i+1:]:
                    if frozenset((a,b)) in ignored:continue
                    distances.append((fcl.distance(checker.robot._objs[a]['obj'],checker.robot._objs[b]['obj']),a,b))
            distance,a,b=min(distances)
            rows.append(dict(object=name,reference=c['reference'],condition=c['id'],opening_m=width,
                             minimum_self_clearance_m=distance,body_pair=[a,b],condition_source=str((folder/'conditions.json').relative_to(ROOT))))
    save(OUT/'planner-clearance-audit.json',dict(method='FCL mesh distance at saved endpoint seeds, excluding exactly USD-adjacent rigid-body pairs; open fingers',
        old_planning_buffer_m=.03,new_planning_buffer_m=.005,physical_contact_threshold_unchanged=True,
        geometry_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'rl/assets/garmi.usd',ROOT/'rl/assets/garmi_training.urdf',ROOT/'assets/kitchen/cooked-model.pkl']},
        rows=rows))
    print('SELF_CLEARANCE_RANGE_M',min(r['minimum_self_clearance_m'] for r in rows),max(r['minimum_self_clearance_m'] for r in rows))


if __name__=='__main__':main()
