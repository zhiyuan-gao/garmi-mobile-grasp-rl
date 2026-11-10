"""RoboKudo observations for the two pick-ups in the unchanged household scene."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from backend import HouseholdBackend
from camera import HeadCamera

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('perceived_object_backend', ROOT/'demos/robokudo_bowl/backend.py')
perception = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perception)


class PerceivedHouseholdBackend(HouseholdBackend):
    def __init__(self, *args, **kwargs):
        self.perceived = {}
        self.observations = []
        self.active_perception = None
        self.perception_control = False
        self.perception_fov_degrees = 70.
        super().__init__(*args, **kwargs)
        from performance import optimize_contact_updates
        self.fast_contacts=optimize_contact_updates(self.env.scene)
        (self.out/'performance-settings.json').write_text(json.dumps(dict(contact_sensors=self.fast_contacts,physics_dt=self.env.physics_dt,history_preserved=True,head_camera_on_demand=True),indent=2))
        for _ in range(30):
            self.step_classical(time.monotonic())
        self.sensor = HeadCamera(self)
        self.original_mdp = self.mdp
        self.observed_object = SimpleNamespace(data=perception.ObjectDataView(self))
        self.observed_env = perception.EnvironmentView(self)
        self.audit = (self.out/'policy-inputs.jsonl').open('w', buffering=1)
        # Both objects are visible from the initial counter approach region.
        # The detector receives RGB-D, calibration and known models only.
        self.capture_object('bowl-initial', 'bowl')
        self.capture_object('mug-initial', 'mug')
        self.use_perception('bowl-initial')

    def capture_object(self, label, obj):
        folder = self.out/'perception'/label
        camera=self.sensor.camera
        camera._render_product.hydra_texture.set_updates_enabled(True)
        if camera.is_paused():
            camera.resume()
        try:
            frame = self.sensor.capture(folder)
        finally:
            camera.pause()
            camera._render_product.hydra_texture.set_updates_enabled(False)
        command = ['bash', '-c',
            'source /opt/ros/jazzy/setup.bash; source /home/jovyan/cram-vrb-lab/ros2_ws/install/setup.bash; '
            'export PYTHONPATH=/workspace/cram-vrb-lab:$PYTHONPATH; export OMP_NUM_THREADS=2; exec "$@"',
            'robokudo', '/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python',
            str(ROOT/'demos/robokudo_bowl/detect.py'), str(frame)]
        if obj == 'mug':
            model = dict(self.mug_metadata)
            model['mesh'] = str(self.mug_workspace/model['mesh'].removeprefix('/workspace/'))
            model_path = folder/'model.json'
            model_path.write_text(json.dumps(model))
            command += ['--model-spec', str(model_path)]
        excluded = ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','ISAAC_PATH','CARB_APP_PATH','EXP_PATH','CONDA_PREFIX','PYTHONEXE')
        with (folder/'detector.log').open('w') as log:
            subprocess.run(command, check=True, timeout=90, stdout=log, stderr=subprocess.STDOUT,
                env={k:v for k,v in os.environ.items() if k not in excluded})
        estimate = json.loads((folder/'estimate.json').read_text())
        self.perceived[label] = estimate
        self.observations.append(dict(id=label, object=obj,
            path='sim/perception/'+label, captured_at_utc=json.loads((folder/'camera.json').read_text())['captured_at_utc']))
        path = self.out/'perception/index.next.json'
        path.write_text(json.dumps(self.observations, indent=2))
        path.replace(self.out/'perception/index.json')
        self.record('perception_ready', observation=label, object_name=obj,
            position_source=estimate['position_source'], position_map=estimate['position_map'],
            surface_fit_rmse_m=estimate['surface_fit_rmse_m'])
        return estimate

    def use_perception(self, label):
        self.active_perception = label
        estimate = self.perceived[label]
        self.estimated_position = torch.tensor([self.frame.position_to_task(estimate['position_map']).tolist()],
            device=self.env.device, dtype=torch.float32)
        self.estimated_rotation = torch.tensor([self.frame.quaternion_to_task(estimate['quaternion_wxyz']).tolist()],
            device=self.env.device, dtype=torch.float32)
        self.mdp = perception.ObservedMDP(self.original_mdp, self.observed_env)
        self.env.fixed_finish.env = self.observed_env
        self.perception_control = True
        self.held_offset = None
        # Reuse the established isolation test, with per-observation evidence.
        self.output = self.out/'perception'/label
        perception.PerceivedBowlBackend.verify_input_isolation(self)

    def estimated_pose(self):
        q = self.frame.quaternion_to_map(self.estimated_rotation[0].cpu().numpy())
        return self.frame.position_to_map(self.estimated_position[0].cpu().numpy()).tolist(), q.tolist()

    def snapshot(self):
        state = super().snapshot()
        if self.perception_control and self.active_perception:
            pos, q = self.estimated_pose()
            if self.active_perception.startswith('bowl'):
                state.update(object_map_position=pos, object_map_quaternion_wxyz=q,
                    object_map_euler=Rotation.from_quat([*q[1:],q[0]]).as_euler('xyz').tolist(), object_velocity=[0.]*6)
            else:
                state.update(mug_map_position=pos, mug_map_quaternion_wxyz=q)
        elif 'mug-initial' in self.perceived and self.stage in ('bowl_counter_to_table','return_to_counter_mug'):
            e = self.perceived['mug-initial']
            state.update(mug_map_position=e['position_map'], mug_map_quaternion_wxyz=e['quaternion_wxyz'])
        # Initial mug localization also feeds the live CRAM world during bowl grasp.
        if self.active_perception == 'bowl-initial' and 'mug-initial' in self.perceived:
            e = self.perceived['mug-initial']
            state.update(mug_map_position=e['position_map'], mug_map_quaternion_wxyz=e['quaternion_wxyz'])
        state['perception_observation'] = self.active_perception
        state['grasp_pose_source'] = 'robokudo_rgbd_model_fit' if self.perception_control else 'simulation_transport_state'
        return state

    def step_policy(self):
        if not self.perception_control:
            raise RuntimeError('A household RL grasp requires a RoboKudo observation')
        phase = int(self.env.fixed_finish.phase[0])
        if phase >= 3:
            from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
            hp,hq,_,_ = self.original_mdp.geometry(self.observed_env)
            if self.held_offset is None:
                self.held_offset = subtract_frame_transforms(hp,hq,self.estimated_position,self.estimated_rotation)
            self.estimated_position[:],self.estimated_rotation[:] = combine_frame_transforms(hp,hq,*self.held_offset)
        obs = self.mdp.observation(self.env)
        self.audit.write(json.dumps(dict(step=self.control_step,phase=phase,observation_id=self.active_perception,
            object_position_task=self.estimated_position[0].cpu().tolist(),observation=obs[0].cpu().tolist()))+'\n')
        return super().step_policy()

    def return_to_cram(self):
        super().return_to_cram()
        # Preserve the existing physical transport/placement feedback. This does
        # not substitute simulated object poses into either RL grasp.
        self.perception_control = False

    def select_mug(self):
        # The base and arms are now back at the existing mug approach station.
        # Reobserve; do not reuse the initial snapshot after the bowl transport.
        self.capture_object('mug-pregrasp', 'mug')
        super().select_mug()
        self.use_perception('mug-pregrasp')

    def hold(self):
        super().hold()
        if self.active_perception == 'mug-pregrasp' and self.last_reason == 'success':
            self.perception_control = False

    def close(self):
        if hasattr(self,'audit'):
            self.audit.close()
        super().close()
