"""Pinch the outer handle bar with one finger inside its hole and one outside.

Object/world Z is up. Approach from either face of the handle plane; closing
axis is radial, unlike the obsolete local-Z-roll experiment. No learned scores.
"""
import argparse,itertools,json
from collections import Counter
import numpy as np,trimesh,yaml
from scipy.spatial.transform import Rotation
from trimesh.collision import CollisionManager
from select_references import ROOT,HAND_OFFSET,sha,farthest_selection
from prepare_rl_validation import hand_parts,save
from rebalance_mug import FingerContacts
from audit_pitcher_contacts import cooked_boundary
from try_handle_roll import host,read,prepare_mask
OUT=ROOT/'runs/ycb-handle-inner-outer-v2'

def main(name):
 source=ROOT/f'resources/validated-ycb-v{5 if name=="pitcher" else 4}/{name}'
 out=OUT/name;out.mkdir(parents=True,exist_ok=True)
 bank=read(source/'bank.json');obj=bank['objects'][0]
 mesh=trimesh.load(host(obj['mesh']),force='mesh',process=False)
 boundary=cooked_boundary(host(obj['cooked']));boundary.export(out/'cooked-boundary.obj')
 d=np.array([.071,.069,0.]) if name=='pitcher' else np.array([1.,0,0]);d/=np.linalg.norm(d)
 n=np.cross([0,0,1],d);tangent=.03 if name=='pitcher' else .015
 heights=np.linspace(.115,.19,7) if name=='pitcher' else np.linspace(.028,.055,7)
 lines=trimesh.intersections.mesh_plane(boundary,n,tangent*n)
 stations=[]
 for height in heights:
  a,b=lines[:,0],lines[:,1];mask=(a[:,2]-height)*(b[:,2]-height)<0;a,b=a[mask],b[mask]
  points=a+(b-a)*((height-a[:,2])/(b[:,2]-a[:,2]))[:,None]
  radii=np.sort(points@d);inner,outer=radii[-2:];body=radii[-3]
  assert inner>body and outer>inner
  stations.append(dict(z=float(height),inner=float(inner),outer=float(outer),body=float(body)))
 checkers=[FingerContacts(m) for m in (mesh,boundary)]
 obstacles=CollisionManager();obstacles.add_object('scan',mesh);obstacles.add_object('physics',boundary)
 records=[];counts={};eligible={}
 for opening in [.08,.04,.03,.02]:
  hand=trimesh.util.concatenate(hand_parts(opening));hand.apply_transform(HAND_OFFSET)
  counts[opening]=Counter();eligible[opening]=[]
  for station,side,yaw,radial,depth in itertools.product(stations,[-1,1],[0,15,30,45],[-.006,-.003,0,.003,.006],[-.035,-.025,-.015,-.005,.005]):
   rz=Rotation.from_euler('z',side*yaw,degrees=True).as_matrix()
   x=rz@d;z=rz@(side*n);y=np.cross(z,x);rot=np.column_stack([x,y,z])
   center=((station['inner']+station['outer'])/2+radial)*d+tangent*n+np.array([0,0,station['z']])+depth*z
   pose=np.eye(4);pose[:3,:3]=rot;pose[:3,3]=center-.0934*z
   row=dict(opening_m=opening,side=side,vertical_z_yaw_offset_deg=side*yaw,radial_offset_m=radial,insertion_offset_m=depth,
            section=station,contact_center_m=center.tolist(),mesh_T_gripper=pose.tolist())
   v=trimesh.transform_points(hand.vertices,pose)
   reason=None
   if min(v[:,2].min(),v[:,2].min()-.08*pose[2,2])-mesh.bounds[0,2]<.002:reason='table_collision'
   if reason is None:
    for distance in np.linspace(0,.08,17):
     pre=pose.copy();pre[:3,3]-=distance*z
     if obstacles.in_collision_single(hand,transform=pre):reason='open_hand_or_approach_collision';break
   if reason is None:
    hits=[c.query(pose,opening_m=opening) for c in checkers];row['contacts']=dict(scan=hits[0],cooked=hits[1])
    if any(c is None for h in hits for c in h):reason='missing_contacts'
    elif any(c['q_m']>=opening/2-1e-5 for h in hits for c in h):reason='initial_contact'
    elif any(abs(a['q_m']-b['q_m'])>.002 for a,b in zip(*hits)):reason='surface_disagreement'
    else:
     # Verify contacts are on radially opposite faces of the outer bar, not
     # on the two lateral sides or on the container body.
     for hit in hits:
      r=sorted(float(np.array(c['point_m'])@d) for c in hit)
      if not (r[0]>station['body']+.003 and r[0]<=station['inner']+.004 and r[1]>=station['outer']-.004 and r[1]-r[0]>.004):
       reason='not_inner_outer_handle_contacts';break
   row['result']=reason or 'geometry_pass';counts[opening][row['result']]+=1
   if reason is None:eligible[opening].append(len(records))
   records.append(row)
  print(name,'opening',opening,dict(counts[opening]),flush=True)
 save(out/'geometry-report.json',dict(object=name,status='geometry_screened',stations=stations,counts_by_opening_m={str(k):dict(v) for k,v in counts.items()},records=records,
  target='Object Z vertical. Approach 0 to 45 degrees from handle plane normal; radial closing axis. One finger in handle hole, other outside outer bar.',
  sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in (source/'bank.json',host(obj['mesh']),host(obj['cooked']),ROOT/'graspgen/try_handle_inner_outer_oblique.py')},
  limitations='Static independent first contact does not prove physical force closure; Horizontal approach 0 to 45 degrees from the handle plane normal; additional shallow insertion search.'))
 opening=max(eligible,key=lambda w:len(eligible[w]));pool=eligible[opening]
 if not pool:return
 poses=np.array([records[i]['mesh_T_gripper'] for i in pool])
 quality=-np.array([abs(records[i]['contacts']['cooked'][0]['q_m']-records[i]['contacts']['cooked'][1]['q_m']) for i in pool])
 selected=farthest_selection(poses,quality,min(12,len(pool)))
 refs=[]
 for j,i in enumerate(selected):
  r=records[pool[i]];h=np.array(r['mesh_T_gripper'])@HAND_OFFSET
  refs.append(dict(id=f'{name}-innerouter-oblique-{j:03d}',position=h[:3,3].tolist(),quaternion=Rotation.from_matrix(h[:3,:3]).as_quat()[[3,0,1,2]].tolist(),
    **{k:v for k,v in r.items() if k!='result'}))
 obj['references']=refs;bank['status']='diagnostic_inner_outer_handle_candidates'
 save(out/'bank.json',bank);save(out/'references.json',refs)
 prepare_mask(source,out,bank,mesh,hand_parts(opening))
 cfg=yaml.safe_load((out/'task.yaml').read_text());cfg['garmi_task']['controller']['gripper_open_width_m']=opening
 (out/'task.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 save(out/'selection.json',dict(opening_m=opening,selected=len(refs),geometry_eligible=len(pool),score_source='None: deterministic geometric candidates; no GraspGen inference or inherited scores.'))
 print('READY',name,len(refs),'width',opening,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--object',required=True,choices=['pitcher','mug']);main(p.parse_args().object)
