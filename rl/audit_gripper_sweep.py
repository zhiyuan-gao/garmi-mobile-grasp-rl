"""Check nominal open-finger/cup intersections omitted by grasp contact rules."""
import argparse
import json
import pickle
from pathlib import Path

import fcl
import numpy as np
from scipy.spatial.transform import Rotation


def convex(mesh):
    mesh=mesh.convex_hull
    faces=np.column_stack((np.full(len(mesh.faces),3),mesh.faces)).astype(np.int32).ravel()
    return fcl.Convex(np.asarray(mesh.vertices),len(mesh.faces),faces)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plans',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    pieces=pickle.load((root/'assets/cup/cooked-model.pkl').open('rb'))
    cup=[fcl.CollisionObject(convex(mesh)) for record,mesh in pieces if record['parent']=='Object']
    fingers={name:[fcl.CollisionObject(convex(mesh)) for record,mesh in pieces if record['body']==name]
             for name in ('right_fr3_leftfinger','right_fr3_rightfinger')}
    plans=json.loads(Path(args.plans).read_text())['results']
    queries=json.loads(Path(args.plans).with_name('initial-queries.json').read_text())
    rows=[]
    for index,(plan,query) in enumerate(zip(plans,queries)):
        if not plan['admissible']:
            continue
        obj=np.array(query['snapshot']['object_state'])
        transform=fcl.Transform(Rotation.from_quat(obj[[4,5,6,3]]).as_matrix(),obj[:3])
        for shape in cup:shape.setTransform(transform)
        collisions=[]
        for tick,node in enumerate(plan['trajectory']):
            hand=np.array(node['hand'])
            for name,shapes in fingers.items():
                offset=np.eye(4)
                sign=1 if 'leftfinger' in name else -1
                offset[:3,3]=[0,sign*.04,.0584]
                if sign==-1:offset[:3,:3]=Rotation.from_euler('z',np.pi).as_matrix()
                pose=hand@offset
                hit=False
                for shape in shapes:
                    shape.setTransform(fcl.Transform(pose[:3,:3],pose[:3,3]))
                    for target in cup:
                        if fcl.collide(shape,target,fcl.CollisionRequest(),fcl.CollisionResult()):
                            hit=True
                            break
                    if hit:break
                if hit:
                    collisions.append(dict(tick=tick+1,seconds=(tick+1)*query['step_dt'],
                        finger=name,stage=node['stage'],distance=node['distance']))
        rows.append(dict(case=index,reference=plan['reference'],collisions=collisions))
    Path(args.out).write_text(json.dumps(dict(status='complete',
        method='50 ms nominal Giskard samples, open fingers, original stationary cup and cooked convex parts',results=rows),indent=2))
    print([(r['case'],len(r['collisions'])) for r in rows])


if __name__=='__main__':main()
