"""Continuous physical mug-body RL grasp and dual-arm drawer manipulation demo."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--case',default='position-1-body-16');p.add_argument('--grasp-only',action='store_true');p.add_argument('--debug-resume');p.add_argument('--debug-stage',choices=['all','close'],default='all')
p.add_argument('--full-scene',action='store_true');p.add_argument('--camera-eye',type=float,nargs=3,default=(-1.4,2.4,2.0));p.add_argument('--camera-target',type=float,nargs=3,default=(.65,.3,.85));p.add_argument('--grasp-camera-eye',type=float,nargs=3);p.add_argument('--grasp-camera-target',type=float,nargs=3);p.add_argument('--preview-only',action='store_true');p.add_argument('--preview-state');AppLauncher.add_app_launcher_args(p);args=p.parse_args()
if args.debug_stage!='all' and not args.debug_resume:p.error('--debug-stage requires --debug-resume')
if args.preview_state and not args.preview_only:p.error('--preview-state requires --preview-only')
args.enable_cameras=True
app=AppLauncher(args).app
import sys,json,math,time,hashlib
from pathlib import Path
sys.path.insert(0,'/demo')
import numpy as np,torch,yaml,imageio.v2 as imageio
from scipy.spatial.transform import Rotation,Slerp
from PIL import Image,ImageDraw,ImageFont
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner
from garmi_rl.config import configured_env
from kinematics import GarmiKinematics,transform,pose_error
out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
(out/'record_mug_drawer.py').write_bytes(Path(__file__).read_bytes())
profile=yaml.safe_load(Path('/workspace/runs/mug-physical-success-review/ppo.yaml').read_text())
checkpoint=Path('/workspace/runs/mug-ppo-physical-success/ppo-6500/checkpoints/update_6500.pt')
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='b6ae2305fcbb353d88801ca3f9817b74665f34f9d8238b6fcd5a96d41e082ce2'
case=next(c for c in json.loads(Path('/demo/configs/mug6500-paired-cases.json').read_text()) if c['id']==args.case);assert case['grasp_type']=='body'
(out/'case.json').write_text(json.dumps(case,indent=2))
# The sampled reset is immediately replaced by the validated demonstration case.
# Avoid loading the unused expert training reset pool during inference.
profile['garmi_task']['multi_object'].pop('training_reset_pool',None)
cfg,task=configured_env(profile,1,20261022)
cfg.scene.robot.spawn.usd_path='/demo/assets/mug-drawer/garmi_dual_preserved.usda'
park=[0.,-.785,0.,-2.356,0.,1.571,.785]
cfg.scene.robot.init_state.joint_pos.update({f'left_fr3_joint{i+1}':v for i,v in enumerate(park)})
cfg.scene.robot.init_state.joint_pos['left_fr3_finger_joint.*']=.04
cfg.scene.robot.actuators['left_arm']=ImplicitActuatorCfg(joint_names_expr=['left_fr3_joint[1-7]'],effort_limit_sim={'left_fr3_joint[1-4]':87.,'left_fr3_joint[5-7]':12.},velocity_limit_sim=1.,stiffness=10000.,damping=200.)
cfg.scene.robot.actuators['left_fingers']=ImplicitActuatorCfg(joint_names_expr=['left_fr3_finger_joint.*'],effort_limit_sim=40.,velocity_limit_sim=.04,stiffness=2000.,damping=100.)
from isaaclab.sensors import ContactSensorCfg
for i in range(1,8):
 setattr(cfg.scene,f'left_contact_{i}',ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/Robot/left_fr3_link{i}',filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/right_fr3_link0','{ENV_REGEX_NS}/Table','{ENV_REGEX_NS}/Drawer','{ENV_REGEX_NS}/Robot/base_link'],history_length=1))
for i,name in enumerate(['left_fr3_leftfinger','left_fr3_rightfinger'],1):
 setattr(cfg.scene,f'left_finger_{i}',ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/Robot/{name}',filter_prim_paths_expr=['{ENV_REGEX_NS}/Drawer'],history_length=1))
cfg.scene.table.spawn.usd_path='/demo/assets/mug-drawer/kitchen.usda'
cfg.scene.drawer=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Drawer',spawn=sim.UsdFileCfg(usd_path='/demo/assets/mug-drawer/drawer.usda',activate_contact_sensors=True),init_state=RigidObjectCfg.InitialStateCfg(pos=(1.053045,.39,.8)))
if args.full_scene:
 cfg.scene.table.spawn.usd_path='/demo/assets/mug-drawer/apartment.usda'
 cfg.scene.drawer.spawn.usd_path='/demo/assets/mug-drawer/drawer_visual.usda'
cfg.viewer.resolution=(1600,900);cfg.viewer.eye=tuple(args.grasp_camera_eye or args.camera_eye);cfg.viewer.lookat=tuple(args.grasp_camera_target or args.camera_target);cfg.sim.render_interval=cfg.decimation
writer=None;env=None;report={'status':'running','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'events':[],'controller':'PPO grasp; scripted Cartesian IK after grasp; passive drawer; contact-only object support'}
report.update(full_apartment=args.full_scene,camera_eye=args.camera_eye,camera_target=args.camera_target,grasp_camera_eye=args.grasp_camera_eye,grasp_camera_target=args.grasp_camera_target)
try:
 env=ManagerBasedRLEnv(cfg,render_mode='rgb_array');env.forbidden_sensors=cfg.forbidden_sensors
 from pxr import UsdGeom,UsdPhysics
 if args.full_scene:
  # Hide the simulator's grid floor appearance; keep its collision plane.
  ground=env.sim.stage.GetPrimAtPath('/World/Ground')
  UsdGeom.Imageable(ground).CreateVisibilityAttr('invisible')
 for prim in env.sim.stage.Traverse():
  if not args.full_scene and '/Table/collider_' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
   UsdGeom.Imageable(prim).CreateVisibilityAttr('inherited');UsdGeom.Imageable(prim).CreatePurposeAttr('default')
 wrapped=SkrlVecEnvWrapper(env);profile['agent']['experiment'].update(directory=str(out),experiment_name='replay',write_interval=0)
 agent=Runner(wrapped,profile).agent;agent.load(str(checkpoint));agent.set_running_mode('eval')
 env.reset();env.grasp_bank.restore([case]);env._reset_idx=lambda ids:None
 r=env.scene['robot'];o=env.scene['object'];drawer=env.scene['drawer']
 if args.preview_only:
  if args.preview_state:
   state=torch.load(args.preview_state,map_location=env.device,weights_only=False)
   for obj,key in [(r,'robot'),(o,'object'),(drawer,'drawer')]:
    obj.write_root_pose_to_sim(state[key][:,:7]);obj.write_root_velocity_to_sim(state[key][:,7:])
   r.write_joint_state_to_sim(state['joints'],state['vel']);env.sim.forward();env.scene.update(0.)
  env.render()
  for _ in range(24):env.sim.render()
  for name,eye,target in [('requested',args.camera_eye,args.camera_target),('counter-left',(1.3,1.6,2.4),(.55,.2,1.)),('counter-right',(1.3,-1.3,2.4),(.55,.2,1.)),('overhead',(.7,.5,2.5),(.6,.2,1.)),('high-left',(-.5,1.7,2.3),(.65,.28,.85)),('front-left',(-1.4,1.7,2.),(.65,.22,.9))]:
   env.sim.set_camera_view(eye,target)
   for _ in range(12):env.sim.render()
   imageio.imwrite(out/f'view-{name}.png',env.render().copy())
  report['status']='preview_completed';raise SystemExit(0)
 ids={s:r.find_joints(f'{s}_fr3_joint[1-7]')[0] for s in ['left','right']};fids={s:r.find_joints(f'{s}_fr3_finger_joint.*')[0] for s in ids}
 print('JOINTS',r.joint_names,'BODIES',r.body_names,flush=True)
 kin={};urdf=Path('/demo/assets/mug-drawer/garmi_dual.urdf').read_text()
 for side in ids:
  path=out/f'{side}-kinematics.urdf';path.write_text(urdf if side=='right' else urdf.replace('right_fr3','temporary_fr3').replace('left_fr3','right_fr3'))
  kin[side]=GarmiKinematics(path)
 # Initialize the RGB render product before writing the first video frame.
 env.render()
 for _ in range(24):env.sim.render()
 writer=imageio.get_writer(out/'raw.mp4',fps=20,codec='libx264',quality=7,macro_block_size=2)
 trace=(out/'trace.jsonl').open('w',buffering=1);steps=0;label='Grasp the mug by its body. | RL policy'
 def snapshot():
  return dict(step=steps,sim_seconds=steps*env.step_dt,mug=o.data.root_pos_w[0].cpu().tolist(),drawer=drawer.data.root_pos_w[0].cpu().tolist(),base=r.data.root_pos_w[0].cpu().tolist(),mug_quaternion_wxyz=o.data.root_quat_w[0].cpu().tolist(),left_fingers=r.data.joint_pos[0,fids['left']].cpu().tolist(),left_finger_forces=[float(env.scene[f'left_finger_{i}'].data.force_matrix_w.norm(dim=-1).max()) for i in (1,2)],right_finger_forces=[float(env.scene[f'finger_{i}'].data.force_matrix_w.norm(dim=-1).max()) for i in (1,2)])
 def capture():
  global steps
  steps+=1
  pic=Image.fromarray(env.render().copy());d=ImageDraw.Draw(pic);d.rectangle((0,0,pic.width,46),fill='black');d.text((18,9),label,fill='white',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26));writer.append_data(np.asarray(pic))
  trace.write(json.dumps(snapshot())+'\n')
  if steps%50==0:pic.save(out/'latest.jpg')
 def event(name,**kw):
  e=dict(event=name,**snapshot(),**kw);report['events'].append(e);(out/'report.json').write_text(json.dumps(report,indent=2));print('EVENT',json.dumps(e),flush=True)
  imageio.imwrite(out/f'{steps:05}-{name}.png',env.render().copy())
  if 'targets' in globals():
   torch.save(dict(robot=r.data.root_state_w,joints=r.data.joint_pos,vel=r.data.joint_vel,object=o.data.root_state_w,drawer=drawer.data.root_state_w,targets=targets,right_target=targets[:,ids['right']],grasp_status='success'),out/f'{steps:05}-{name}.pt')
 saved=None
 if args.debug_resume:
  saved=torch.load(args.debug_resume,map_location=env.device,weights_only=False)
  if 'grasp_status' not in saved:
   previous=json.loads(Path(args.debug_resume).with_name('report.json').read_text())
   saved['grasp_status']=next(e['status'] for e in previous['events'] if e['event']=='rl_grasp')
   saved['right_target']=saved['joints'][:,ids['right']]
  if saved.get('grasp_status')!='success':raise ValueError('Resume requires a verified grasp state')
  for obj,key in [(r,'robot'),(o,'object'),(drawer,'drawer')]:
   obj.write_root_pose_to_sim(saved[key][:,:7]);obj.write_root_velocity_to_sim(saved[key][:,7:])
  r.write_joint_state_to_sim(saved['joints'],saved['vel']);r.set_joint_position_target(saved['joints']);env.sim.forward();env.scene.update(0.)
  status='success';report['debug_resumed']=True
 else:
  obs=env.observation_manager.compute()['policy'];status='timeout'
  for step in range(math.ceil(cfg.episode_length_s/env.step_dt)+2):
   with torch.no_grad():
    sampled,_,extra=agent.act(obs,timestep=step,timesteps=800);obs,_,done,truncated,_=wrapped.step(extra.get('mean_actions',sampled))
   capture()
   if bool((done|truncated).flatten()[0]):
    d=env.diagnostics;status=next((k for k in ['collision','dropped','out_of_bounds','success','approach_timeout','attempt_failed'] if k in d and bool(d[k][0])),'timeout');break
 from garmi_rl.contact_policy import collision_forces
 event('rl_grasp',status=status,forbidden_forces=collision_forces(env)[0][0].cpu().tolist(),sensors={k:env.scene[k].cfg.prim_path for k in env.forbidden_sensors})
 torch.save(dict(robot=r.data.root_state_w,joints=r.data.joint_pos,vel=r.data.joint_vel,object=o.data.root_state_w,drawer=drawer.data.root_state_w,grasp_status=status,right_target=env.action_manager.get_term('arm').processed_actions),out/'post-grasp.pt')
 if status!='success':raise RuntimeError('RL grasp: '+status)
 if args.grasp_camera_eye:
  # A camera cut only: the same physical simulation continues without a reset.
  env.sim.set_camera_view(args.camera_eye,args.camera_target)
  for _ in range(12):env.sim.render()
 if args.grasp_only:
  report['status']='grasp_verified'
 else:
  targets=saved['targets'].clone() if saved and 'targets' in saved else r.data.joint_pos.clone();targets[:,ids['right']]=saved['right_target'] if saved else env.action_manager.get_term('arm').processed_actions;targets[:,fids['right']]=0.
  base=env.action_manager.get_term('base');base._processed_actions.zero_();base.velocity.zero_()
  def tick():
   for _ in range(env.cfg.decimation):
    env._sim_step_counter+=1;base.apply_actions();r.set_joint_position_target(targets);env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
   env.common_step_counter+=1;capture()
  def root_pose():
   state=r.data.root_state_w[0].cpu().numpy();return transform(state[:3],Rotation.from_quat(state[[4,5,6,3]]).as_matrix())
  def hand(side):return root_pose()@kin[side].forward(r.data.joint_pos[0,ids[side]].cpu().numpy())
  def move(side,goal,seconds=3.):
   initial=hand(side);N=max(1,round(seconds/env.step_dt));rot=Slerp([0,1],Rotation.from_matrix([initial[:3,:3],goal[:3,:3]]));q=targets[0,ids[side]].cpu().numpy()
   for i in range(N):
    a=(i+1)/N;a=a*a*(3-2*a);target=transform(initial[:3,3]*(1-a)+goal[:3,3]*a,rot(a).as_matrix())
    q,err=kin[side].solve(np.linalg.inv(root_pose())@target,q)
    if np.linalg.norm(err[:3])>.025 or np.linalg.norm(err[3:])>.15:raise RuntimeError(f'{side} IK unreachable: {err}')
    targets[:,ids[side]]=torch.as_tensor(q,device=env.device,dtype=targets.dtype);tick()
   for _ in range(80):
    error=pose_error(goal,hand(side))
    if np.linalg.norm(error[:3])<.012 and np.linalg.norm(error[3:])<.08:break
    tick()
   error=pose_error(goal,hand(side));event('move_'+side,error=error.tolist(),target_joints=targets[0,ids[side]].cpu().tolist(),actual_joints=r.data.joint_pos[0,ids[side]].cpu().tolist(),left_contact_forces={str(i):env.scene[f'left_contact_{i}'].data.force_matrix_w[0].norm(dim=-1).cpu().tolist() for i in range(1,8)})
   if np.linalg.norm(error[:3])>.03 or np.linalg.norm(error[3:])>.2:raise RuntimeError(f'{side} tracking blocked: {error}')
  def wait(seconds):
   for _ in range(round(seconds/env.step_dt)):tick()
  def grip(side,width):targets[:,fids[side]]=width/2;wait(1.)
  if args.debug_stage!='close':
   label='Hold the mug in the right hand.'
   goal=hand('right');goal[:3,3]+=[0.,-.12,.04];move('right',goal,3.);safe_right=hand('right').copy()
   label='Open the drawer with the left hand.'
   # The handle runs along world Y; close the fingers vertically around it.
   rotation=np.array([[0,0,1],[1,0,0],[0,1,0]],float)
   goal=transform([.56,.39,1.12],rotation);move('left',goal,4.);goal[:3,3][2]=.8;move('left',goal,3.)
   goal[:3,3]=[.674,.39,.8];move('left',goal,2.);grip('left',0.)
   event('handle_grasp');goal[:3,3]+=[-.36,0,0];move('left',goal,5.);wait(.5)
   opening=1.053045-float(drawer.data.root_pos_w[0,0]);event('drawer_open',opening=opening)
   if opening<.28:raise RuntimeError('Drawer did not open by physical hand contact')
   if float(o.data.root_pos_w[0,2])<1.:raise RuntimeError('Mug was not retained while opening drawer')
   label='Place the mug inside the drawer.'
   offset=hand('right')[:3,3]-o.data.root_pos_w[0].cpu().numpy()
   center=np.array([1.00-opening,.39,.94]);goal=hand('right');goal[:3,3]=center+offset;move('right',goal,5.)
   goal[:3,3][2]-=.205;move('right',goal,3.);grip('right',.08);wait(1.)
   event('mug_released');goal[:3,3][2]+=.25;move('right',goal,3.);move('right',safe_right,4.);event('right_cleared')
  opening=1.053045-float(drawer.data.root_pos_w[0,0])
  label='Close the drawer with the left hand.'
  goal=hand('left');goal[:3,3][0]+=(opening-.005);move('left',goal,5.);grip('left',.08);goal[:3,3][0]-=.18;move('left',goal,2.);wait(2.)
  mug=o.data.root_pos_w[0].cpu().numpy();opening=1.053045-float(drawer.data.root_pos_w[0,0]);velocity=o.data.root_vel_w[0,:3].norm().item()
  import itertools
  object_meta=json.loads(Path(cfg.multi_object['manifest']).read_text())['objects'][0]
  corners=np.array(list(itertools.product(*zip(*object_meta['bounds']))))
  quat=o.data.root_quat_w[0].cpu().numpy();rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
  points=corners@rotation.T+mug;lo,hi=points.min(0),points.max(0)
  tilt=float(np.arccos(np.clip(rotation[2,2],-1,1)))
  inside=bool(np.all(lo[:2]>=[.830045-opening+.002,.2145+.002]) and np.all(hi[:2]<=[1.294045-opening-.002,.5655-.002]) and lo[2]>=.7245-.002 and hi[2]<=.825+.002 and tilt<np.deg2rad(10))
  report.update(mug_world_bounds=[lo.tolist(),hi.tolist()],mug_tilt_degrees=float(np.rad2deg(tilt)))
  report.update(status='completed' if inside and opening<.02 and velocity<.02 else 'verification_failed',drawer_opening_m=opening,mug_inside=bool(inside),mug_speed=velocity)
  event('final_verification')
 imageio.imwrite(out/'final.png',env.render().copy())
except Exception as exc:
 import traceback;traceback.print_exc();report.update(status='incomplete',error=str(exc))
finally:
 (out/'report.json').write_text(json.dumps(report,indent=2))
 if writer:writer.close()
 if env:env.close()
 app.close()

if report['status'] not in ('completed','grasp_verified'):
 raise SystemExit(1)
