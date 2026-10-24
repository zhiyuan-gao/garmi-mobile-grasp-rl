"""One physical scene for bowl delivery, counter mug grasp and passive drawer use."""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
import yaml

from garmi_cram_rl.lab_backend import LabBackend
from garmi_cram_rl.policy import Actor
from kinematics import GarmiKinematics, transform, pose_error

PARK = [0., -.785398, 0., -2.356194, 0., 1.570796, .785398]
MUG_SHA = 'b6ae2305fcbb353d88801ca3f9817b74665f34f9d8238b6fcd5a96d41e082ce2'


class HouseholdBackend(LabBackend):
    def __init__(self, bundle, assets, mug_workspace, out, render=True, display=False):
        self.assets, self.mug_workspace, self.out = map(Path, (assets, mug_workspace, out))
        self.desktop_display = display
        self.stage = 'bowl_counter_to_table'
        self.sequence_report = {'status': 'running', 'scene_initializations': 1, 'events': []}
        self.mug_profile = yaml.safe_load((self.mug_workspace / 'runs/mug-physical-success-review/ppo.yaml').read_text())
        path = self.mug_workspace / 'resources/validated-ycb-v6/mug/bank.json'
        self.mug_metadata = json.loads(path.read_text())['objects'][0]
        self.mug_reference = next(r for r in self.mug_metadata['references'] if r['id'] == 'mug-v3-ref025')
        self.mug_checkpoint = self.mug_workspace / 'runs/mug-ppo-physical-success/ppo-6500/checkpoints/update_6500.pt'
        if hashlib.sha256(self.mug_checkpoint.read_bytes()).hexdigest() != MUG_SHA:
            raise ValueError('Mug checkpoint differs from the validated update-6500 release')
        display_options = {'display': True} if display else {}
        super().__init__(bundle, [0., 5.8, math.pi/2], [.2, 7.4], render=render,
                         full_scene=True, configure=self.configure_scene, **display_options)
        from pxr import UsdGeom
        UsdGeom.Imageable(self.env.sim.stage.GetPrimAtPath('/World/Ground')).CreateVisibilityAttr('invisible')
        self.mug = self.env.scene['mug']
        self.drawer = self.env.scene['drawer']
        self.ids = {s: self.robot.find_joints(f'{s}_fr3_joint[1-7]')[0] for s in ('left', 'right')}
        self.fids = {s: self.robot.find_joints(f'{s}_fr3_finger_joint.*')[0] for s in self.ids}
        self.kin = {}
        urdf = (self.assets / 'garmi_dual.urdf').read_text()
        for side in self.ids:
            path = self.out / f'{side}-kinematics.urdf'
            path.write_text(urdf if side == 'right' else urdf.replace('right_fr3', 'temporary_fr3').replace('left_fr3', 'right_fr3'))
            self.kin[side] = GarmiKinematics(path)
        self.record('initialized', mug_present_from_start=True, mug_checkpoint_sha256=MUG_SHA)

    def configure_scene(self, cfg):
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sensors import ContactSensorCfg
        import isaaclab.sim as sim
        if self.desktop_display:
            cfg.ui_window_class_type = None
            cfg.viewer.resolution = (640, 360)
            cfg.sim.render_interval = max(1, round(.04 / cfg.sim.dt))
        cfg.scene.robot.spawn.usd_path = str(self.assets / 'garmi_dual_preserved.usda')
        cfg.scene.table.spawn.usd_path = str(self.assets / 'apartment.usda')
        cfg.scene.robot.init_state.joint_pos.update({f'left_fr3_joint{i+1}': q for i, q in enumerate(PARK)})
        cfg.scene.robot.init_state.joint_pos['left_fr3_finger_joint.*'] = .04
        cfg.scene.robot.actuators['left_arm'] = ImplicitActuatorCfg(joint_names_expr=['left_fr3_joint[1-7]'],
            effort_limit_sim={'left_fr3_joint[1-4]':87., 'left_fr3_joint[5-7]':12.}, velocity_limit_sim=1., stiffness=10000., damping=200.)
        cfg.scene.robot.actuators['left_fingers'] = ImplicitActuatorCfg(joint_names_expr=['left_fr3_finger_joint.*'],
            effort_limit_sim=40., velocity_limit_sim=.04, stiffness=2000., damping=100.)
        cfg.scene.drawer = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Drawer',
            spawn=sim.UsdFileCfg(usd_path=str(self.assets/'drawer_visual.usda'), activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(1.053045,.39,.8)))
        # Both objects start on the kitchen counter. Translate the validated mug
        # case sideways so its mesh stays separate from the initial bowl.
        pos = self.frame.position_to_task([-.15, 7.262585296174657, .946535943059])
        cfg.scene.mug = copy.deepcopy(cfg.scene.object)
        cfg.scene.mug.prim_path = '{ENV_REGEX_NS}/Mug'
        cfg.scene.mug.spawn.usd_path = str(self.mug_workspace / self.mug_metadata['usd'].removeprefix('/workspace/'))
        cfg.scene.mug.init_state.pos = tuple(pos)
        cfg.scene.mug.init_state.rot = (.7071067811865476, 0., 0., -.7071067811865476)
        for i in (1, 2):
            sensor = copy.deepcopy(getattr(cfg.scene, f'finger_{i}'))
            sensor.filter_prim_paths_expr = ['{ENV_REGEX_NS}/Mug']
            setattr(cfg.scene, f'mug_finger_{i}', sensor)
        cfg.scene.mug_support = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Mug',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Table', '{ENV_REGEX_NS}/Drawer'], history_length=1)
        cfg.scene.bowl_mug_contact = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Object',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Mug'], history_length=1)
        left = [f'left_fr3_link{i}' for i in range(1,8)] + ['left_fr3_leftfinger','left_fr3_rightfinger']
        self.bowl_forbidden = list(cfg.forbidden_sensors)
        self.mug_forbidden = []
        for key in self.bowl_forbidden:
            sensor = getattr(cfg.scene, key)
            sensor.filter_prim_paths_expr += ['{ENV_REGEX_NS}/Drawer'] + ['{ENV_REGEX_NS}/Robot/'+x for x in left]
            mug_sensor = copy.deepcopy(sensor)
            # For the mug skill the bowl becomes an obstacle, including for fingers.
            mug_sensor.filter_prim_paths_expr = [x.replace('/Object','/Mug') for x in sensor.filter_prim_paths_expr] + ['{ENV_REGEX_NS}/Object']
            sensor.filter_prim_paths_expr += ['{ENV_REGEX_NS}/Mug']
            setattr(cfg.scene, 'mug_'+key, mug_sensor)
            self.mug_forbidden.append('mug_'+key)
        # Left-arm contact monitoring remains active during right-arm RL as well.
        for i, body in enumerate(left):
            key = f'left_forbidden_{i}'
            targets = ['{ENV_REGEX_NS}/Table', '{ENV_REGEX_NS}/Object', '{ENV_REGEX_NS}/Mug', '{ENV_REGEX_NS}/Drawer']
            targets += ['{ENV_REGEX_NS}/Robot/'+x for x in ['base_link']+[f'right_fr3_link{j}' for j in range(8)]+['right_fr3_leftfinger','right_fr3_rightfinger']]
            setattr(cfg.scene, key, ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+body, filter_prim_paths_expr=targets, history_length=10))
            self.bowl_forbidden.append(key)
            self.mug_forbidden.append(key)
        cfg.forbidden_sensors = self.bowl_forbidden

    def snapshot(self):
        state = super().snapshot()
        if hasattr(self, 'mug'):
            state.update(sequence_stage=self.stage, mug_map_position=self.frame.position_to_map(self.mug.data.root_pos_w[0].cpu().numpy()).tolist(),
                         mug_map_quaternion_wxyz=self.frame.quaternion_to_map(self.mug.data.root_quat_w[0].cpu().numpy()).tolist(), drawer_opening_m=self.opening())
        return state

    def opening(self):
        return 1.053045-float(self.drawer.data.root_pos_w[0,0])

    def record(self, event, **values):
        entry = dict(event=event, stage=self.stage, simulation_seconds=self.control_step*self.env.step_dt,
                     bowl_map=self.frame.position_to_map(self.object.data.root_pos_w[0].cpu().numpy()).tolist(),
                     mug_map=self.frame.position_to_map(self.mug.data.root_pos_w[0].cpu().numpy()).tolist(),
                     base_map=self.snapshot()['base_map_position'], drawer_opening_m=self.opening(), **values)
        self.sequence_report['events'].append(entry)
        path = self.out/'sequence.next.json'
        path.write_text(json.dumps(self.sequence_report, indent=2)+'\n')
        path.replace(self.out/'sequence.json')
        print('SEQUENCE_EVENT', json.dumps(entry), flush=True)

    def tick(self):
        self._physics(policy=False)
        yield

    def wait(self, seconds):
        for _ in range(round(seconds/self.env.step_dt)):
            yield from self.tick()

    def root_pose(self):
        state = self.robot.data.root_state_w[0].cpu().numpy()
        return transform(state[:3], Rotation.from_quat(state[[4,5,6,3]]).as_matrix())

    def hand(self, side):
        return self.root_pose() @ self.kin[side].forward(self.robot.data.joint_pos[0,self.ids[side]].cpu().numpy())

    def move(self, side, goal, seconds=3.):
        initial = self.hand(side)
        count = max(1, round(seconds/self.env.step_dt))
        rot = Slerp([0,1], Rotation.from_matrix([initial[:3,:3], goal[:3,:3]]))
        q = self.targets[0,self.ids[side]].cpu().numpy()
        for i in range(count):
            a = (i+1)/count; a = a*a*(3-2*a)
            target = transform(initial[:3,3]*(1-a)+goal[:3,3]*a, rot(a).as_matrix())
            q, error = self.kin[side].solve(np.linalg.inv(self.root_pose())@target, q)
            if np.linalg.norm(error[:3])>.025 or np.linalg.norm(error[3:])>.15:
                raise RuntimeError(f'{side} IK unreachable: {error}')
            self.targets[:,self.ids[side]] = self.targets.new_tensor(q)
            yield from self.tick()
        for _ in range(80):
            error = pose_error(goal,self.hand(side))
            if np.linalg.norm(error[:3])<.012 and np.linalg.norm(error[3:])<.08:
                break
            yield from self.tick()
        error = pose_error(goal,self.hand(side))
        self.record('arm_motion', side=side, error=error.tolist())
        if np.linalg.norm(error[:3])>.03 or np.linalg.norm(error[3:])>.2:
            raise RuntimeError(f'{side} tracking blocked: {error}')

    def navigate(self, goal, timeout=40.):
        """Low-speed waypoint servo, using the same acceleration-limited base action."""
        self.record('base_motion_start', goal=goal)
        for _ in range(round(timeout/self.env.step_dt)):
            state = self.snapshot(); xy=np.asarray(state['base_map_position'][:2]); yaw=state['base_map_yaw']
            delta=np.asarray(goal[:2])-xy; angle=(goal[2]-yaw+np.pi)%(2*np.pi)-np.pi
            if np.linalg.norm(delta)<.012 and abs(angle)<.012:
                break
            world=np.clip(1.5*delta,-.18,.18)
            local=Rotation.from_euler('z',-yaw).apply([*world,0])[:2]
            self.base_term._processed_actions[:] = self.base_term.velocity.new_tensor([*local,np.clip(1.5*angle,-.3,.3)])
            yield from self.tick()
        else:
            raise TimeoutError(f'Base waypoint not reached: {goal}')
        self.base_term._processed_actions.zero_()
        yield from self.wait(.6)
        self.record('base_motion_done', goal=goal)

    def select_mug(self):
        env=self.env
        # Swap semantic handles only. Both original rigid bodies remain in the scene.
        env.scene._rigid_objects['object'], env.scene._rigid_objects['mug'] = self.mug, self.object
        for key in ('finger_1','finger_2','support'):
            env.scene._sensors[key], env.scene._sensors['mug_'+key] = env.scene._sensors['mug_'+key],env.scene._sensors[key]
        task=self.mug_profile['garmi_task']
        env.cfg.contact_policy=task['contact_policy'];env.cfg.require_strict_success=task['require_strict_success']
        env.cfg.fixed_finish.update(task['controller']);env.fixed_finish.settings=env.cfg.fixed_finish
        env.cfg.episode_length_s=task['episode_length_s']
        env.cfg.observation_time_scale_s=task['observation_time_scale_s']
        env.cfg.kitchen=copy.deepcopy(env.cfg.kitchen)
        env.cfg.kitchen['movement_bounds_task_xy']=copy.deepcopy(task['kitchen']['movement_bounds_task_xy'])
        env.cfg.kitchen['drop_height_m']=task['kitchen']['drop_height_m']
        env.forbidden_sensors=self.mug_forbidden
        wrist=next(k for k in self.mug_forbidden if env.scene[k].cfg.prim_path.endswith('/right_fr3_link7'))
        paths=env.scene[wrist].cfg.filter_prim_paths_expr
        env._wrist_target_filter=(wrist,next(i for i,p in enumerate(paths) if p.endswith('/Mug')),len(paths))
        env.action_manager.reset();env.fixed_finish.reset(torch.tensor([0],device=env.device));env.episode_length_buf.zero_()
        self.base_term.velocity.zero_()
        env.grasp_pos_o[:]=env.grasp_pos_o.new_tensor(self.mug_reference['position'])
        env.grasp_quat_o[:]=env.grasp_quat_o.new_tensor(self.mug_reference['quaternion'])
        env.grasp_open_width[:]=.08
        for name in ('hold','ever_lifted','anchor','held_last'):
            env.grasp_state[name].zero_()
        env.grasp_state['initial_z'][:]=self.mug.data.root_pos_w[:,2]
        env.diagnostic_step=-1
        self.policy=Actor(torch.load(self.mug_checkpoint,map_location='cpu',weights_only=False)).eval()
        self.reference=None
        self.last_reason=None
        self.record('mug_policy_selected', reference=self.mug_reference['id'], initial_height=float(env.grasp_state['initial_z'][0]))

    def grip(self, side, width):
        self.targets[:,self.fids[side]]=width/2
        yield from self.wait(1.)

    def continue_with_mug(self):
        self.clear_classical_commands();self.base_term._processed_actions.zero_()
        self.record('bowl_segment_verified')
        self.bowl_delivery_position=self.object.data.root_pos_w[0].cpu().numpy().copy()
        self.stage='return_to_counter_mug'
        self.camera([-1.2,4.4,2.7],[.7,6.3,1.])
        for goal in ([.25,6.,math.pi/2],[-.0547544,6.,math.pi/2],[-.0547544,6.3517334,math.pi/2]):
            yield from self.navigate(goal)
        self.stage='mug_rl_approach'
        mug_map=self.frame.position_to_map(self.mug.data.root_pos_w[0].cpu().numpy())
        self.camera(mug_map+[-.55,.28,.85],mug_map+[0.,0.,.08])
        self.select_mug()
        for _ in range(math.ceil(self.env.cfg.episode_length_s/self.env.step_dt)+2):
            with torch.inference_mode():
                result=self.step_policy()
            yield
            if result:
                self.record('mug_rl_result', result=result, diagnostics=self.last_measurements, distance=float(self.env.diagnostics['distance'][0]), angle=float(self.env.diagnostics['angle'][0]), forces=self.env.diagnostics['forbidden_force'][0].cpu().tolist(), sensors=self.mug_forbidden)
                if result!='success':raise RuntimeError('Mug RL grasp: '+result)
                break
        else:raise TimeoutError('Mug RL approach did not finish')
        self.hold()
        self.stage='carry_mug_to_drawer'
        self.camera([-1.2,4.4,2.7],[.7,6.3,1.])
        # Undo only the initial sideways translation of the validated mug case.
        # Its RL policy turns the base during grasping. Preserve that measured
        # heading and arm posture, as the standalone drawer demo does.
        state = self.snapshot()
        goal = [state['base_map_position'][0] + .3073564969799577,
                state['base_map_position'][1], state['base_map_yaw']]
        yield from self.navigate(goal)
        self.stage='open_drawer_left'
        self.env.sim.set_camera_view((-.65,2.,2.4),(.5,.2,.95))
        goal=self.hand('right');goal[:3,3]+=[0.,-.12,.04]
        yield from self.move('right',goal,3.)
        safe_right=self.hand('right').copy()
        rotation=np.array([[0,0,1],[1,0,0],[0,1,0]],float)
        goal=transform([.56,.39,1.12],rotation)
        yield from self.move('left',goal,4.)
        goal[2,3]=.8;yield from self.move('left',goal,3.)
        goal[:3,3]=[.674,.39,.8];yield from self.move('left',goal,2.)
        yield from self.grip('left',0.)
        goal[:3,3]+=[-.36,0,0];yield from self.move('left',goal,5.)
        yield from self.wait(.5)
        opening=self.opening();self.record('drawer_open',opening=opening)
        if opening<.28:raise RuntimeError('Drawer did not open by physical contact')
        if float(self.mug.data.root_pos_w[0,2])<1.:raise RuntimeError('Mug was not retained while opening drawer')
        self.stage='place_mug_inside'
        offset=self.hand('right')[:3,3]-self.mug.data.root_pos_w[0].cpu().numpy()
        goal=self.hand('right');goal[:3,3]=np.array([1.-opening,.39,.94])+offset
        yield from self.move('right',goal,5.)
        # Clear the counter lip before descending into the open drawer.
        # Keep this horizontal segment separate to preserve the approach IK branch.
        goal[0,3]-=.06;yield from self.move('right',goal,2.)
        goal[2,3]-=.205;yield from self.move('right',goal,3.)
        yield from self.grip('right',.08);yield from self.wait(1.)
        self.record('mug_released')
        goal[2,3]+=.25;yield from self.move('right',goal,3.)
        yield from self.move('right',safe_right,4.)
        self.stage='close_drawer_left'
        goal=self.hand('left');goal[0,3]+=self.opening()-.005
        yield from self.move('left',goal,5.)
        yield from self.grip('left',.08)
        goal[0,3]-=.18;yield from self.move('left',goal,2.)
        yield from self.wait(2.)
        self.verify_final()

    def verify_final(self):
        mug=self.mug.data.root_pos_w[0].cpu().numpy()
        quat=self.mug.data.root_quat_w[0].cpu().numpy()
        rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        corners=np.array(list(itertools.product(*zip(*self.mug_metadata['bounds']))))
        points=corners@rotation.T+mug;lo,hi=points.min(0),points.max(0)
        opening=self.opening()
        tilt=float(np.arccos(np.clip(rotation[2,2],-1,1)))
        inside=bool(np.all(lo[:2]>=[.830045-opening+.002,.2145+.002]) and np.all(hi[:2]<=[1.294045-opening-.002,.5655-.002]) and lo[2]>=.7245-.002 and hi[2]<=.825+.002 and tilt<np.deg2rad(10))
        speed=float(self.mug.data.root_lin_vel_w[0].norm())
        bowl_shift=float(np.linalg.norm(self.object.data.root_pos_w[0].cpu().numpy()-self.bowl_delivery_position))
        passed=inside and abs(opening)<.02 and speed<.02 and bowl_shift<.05
        self.stage='completed' if passed else 'verification_failed'
        self.sequence_report.update(status=self.stage,mug_inside=inside,mug_speed=speed,bowl_shift=bowl_shift,drawer_opening_m=opening)
        self.record('final_verification',passed=passed,mug_bounds=[lo.tolist(),hi.tolist()])
        if not passed:raise RuntimeError('Final mug/drawer/bowl verification failed')
