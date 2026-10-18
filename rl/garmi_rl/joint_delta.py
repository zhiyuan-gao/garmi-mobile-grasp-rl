"""Versioned joint increment interface; no inverse kinematics in policy execution."""
import torch
from .expert_actions import PARK, ARM_TARGET_LOWER, ARM_TARGET_UPPER

ACTION_SPACE = 'base_velocity_joint_delta_v1'
COMMAND_VERSION = 'giskard_joint_delta_v1'
JOINT_SCALE = .05  # rad per 50 ms tick, matching the existing 1 rad/s drive limit


def is_joint_delta(profile):
    return profile.get('garmi_task', {}).get('action_space') == ACTION_SPACE


def decode_joint_delta(measured, action):
    return (measured+action.clamp(-1., 1.)*JOINT_SCALE).clamp(
        measured.new_tensor(ARM_TARGET_LOWER), measured.new_tensor(ARM_TARGET_UPPER))


def encode_joint_target(measured, target):
    return ((target-measured)/JOINT_SCALE).clamp(-1., 1.)


class JointPointLoss:
    """GA-DDPG-style gripper landmark loss evaluated by differentiable FK.

    The original method uses Cartesian actions. Here an extra joint-increment
    MSE preserves the seven-joint posture (a 6D hand pose cannot identify it).
    Base-velocity MSE is also a mobile-robot adaptation, not an original term.
    Landmarks: GA-DDPG core/utils.py, revision 14a65e1 (MIT), rotated into the
    GARMI hand convention with finger opening along y. No collision geometry
    is replaced by these six loss landmarks.
    """
    def __init__(self, device):
        from pathlib import Path
        import xml.etree.ElementTree as ET
        import numpy as np
        from scipy.spatial.transform import Rotation
        # Pure forward-kinematic constants, usable during offline BC without
        # launching Isaac Sim or importing any inverse-kinematics controller.
        root = ET.parse(Path(__file__).resolve().parents[1]/'assets/garmi_training.urdf').getroot()
        by_child = {j.find('child').get('link'):j for j in root.findall('joint')}
        chain, name = [], 'right_fr3_hand'
        while name in by_child:
            joint = by_child[name]; chain.append(joint); name = joint.find('parent').get('link')
        self.chain = []
        for joint in reversed(chain):
            origin = joint.find('origin'); transform = np.eye(4)
            transform[:3,3] = np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            transform[:3,:3] = Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            index = next((i for i in range(7) if joint.get('name')==f'right_fr3_joint{i+1}'),None)
            if index is not None:
                assert np.allclose(np.fromstring(joint.find('axis').get('xyz'),sep=' '),[0,0,1])
            self.chain.append((torch.tensor(transform,device=device,dtype=torch.float32),index))
        self.points = torch.tensor([[0.,0.,0.],[0.,0.,0.],
            [0.,-.053,.075],[0.,.053,.075],[0.,-.053,.105],[0.,.053,.105]], device=device)
        self.park = torch.tensor(PARK, device=device, dtype=torch.float32)

    def transform_at(self, joints):
        transform = torch.eye(4, device=joints.device, dtype=joints.dtype).expand(len(joints),-1,-1)
        for origin, index in self.chain:
            transform = transform @ origin.to(dtype=joints.dtype)
            if index is not None:
                c, s = joints[:,index].cos(), joints[:,index].sin()
                zero, one = torch.zeros_like(c), torch.ones_like(c)
                rotation = torch.stack((c,-s,zero,zero,s,c,zero,zero,
                    zero,zero,one,zero,zero,zero,zero,one),dim=-1).reshape(-1,4,4)
                transform = transform @ rotation
        return transform

    def points_at(self, joints):
        transform = self.transform_at(joints)
        return self.points.to(dtype=joints.dtype) @ transform[:,:3,:3].transpose(1,2)+transform[:,None,:3,3]

    def metrics(self, predicted, target, raw_states):
        measured = raw_states[:,:7]+self.park
        # Do not clamp predictions in the loss: clipping would eliminate the
        # restoring gradient for out-of-range actor outputs.
        actual_points = self.points_at(measured+JOINT_SCALE*predicted[:,3:])
        with torch.no_grad():
            expert_points = self.points_at(measured+JOINT_SCALE*target[:,3:])
            current_rotation = self.transform_at(measured)[:,:3,:3]
        # Match the original local-hand frame: L1 distance is frame-dependent.
        pose = ((actual_points-expert_points) @ current_rotation).abs().sum(-1).mean()
        base = (predicted[:,:3]-target[:,:3]).square().mean()
        joint = (predicted[:,3:]-target[:,3:]).square().mean()
        return dict(loss=pose/.01+base+.1*joint,
                    hand_point_l1_m=pose, base_mse=base, joint_delta_mse=joint)

    def __call__(self, predicted, target, raw_states):
        return self.metrics(predicted,target,raw_states)['loss']
