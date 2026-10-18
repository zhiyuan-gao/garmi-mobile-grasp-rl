"""URDF forward kinematics for initializing diagnostic grasp configurations."""
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import quat_from_matrix, compute_pose_error


class DiagnosticIK:
    def __init__(self, device):
        root = ET.parse(Path(__file__).resolve().parents[1]/"assets/garmi_training.urdf").getroot()
        bychild = {j.find("child").get("link"):j for j in root.findall("joint")}
        chain, name = [], "right_fr3_hand"
        while name in bychild:
            joint = bychild[name]
            chain.append(joint)
            name = joint.find("parent").get("link")
        self.chain = []
        for joint in reversed(chain):
            origin = joint.find("origin")
            t = np.eye(4)
            t[:3,3] = np.fromstring(origin.get("xyz","0 0 0"),sep=" ")
            t[:3,:3] = Rotation.from_euler("xyz",np.fromstring(origin.get("rpy","0 0 0"),sep=" ")).as_matrix()
            number = next((i for i in range(7) if joint.get("name") == f"right_fr3_joint{i+1}"), None)
            self.chain.append((torch.tensor(t,device=device,dtype=torch.float32),number))

    def forward(self, q):
        t = torch.eye(4,device=q.device).expand(len(q),-1,-1).clone()
        axes, origins = [], []
        for origin, index in self.chain:
            t = t @ origin
            if index is not None:
                axes.append(t[:,:3,2].clone())
                origins.append(t[:,:3,3].clone())
                rotation = torch.eye(4,device=q.device).expand(len(q),-1,-1).clone()
                c,s = q[:,index].cos(), q[:,index].sin()
                rotation[:,0,0]=c; rotation[:,0,1]=-s
                rotation[:,1,0]=s; rotation[:,1,1]=c
                t = t @ rotation
        p = t[:,:3,3]
        axis = torch.stack(axes,dim=-1)
        offsets = p[:,:,None]-torch.stack(origins,dim=-1)
        jacobian = torch.cat((torch.cross(axis,offsets,dim=1),axis),dim=1)
        return p, quat_from_matrix(t[:,:3,:3]), jacobian

    def solve(self, position, rotation, initial, limits, iterations=200):
        q = initial.clone()
        eye = torch.eye(6,device=q.device).expand(len(q),-1,-1)
        for _ in range(iterations):
            p,r,j = self.forward(q)
            ep,er = compute_pose_error(p,r,position,rotation,rot_error_type="axis_angle")
            error = torch.cat((ep,er),dim=-1)
            step = (j.transpose(1,2) @ torch.linalg.solve(j @ j.transpose(1,2)+.0025*eye,error[:,:,None])).squeeze(-1)
            q = (q+step.clamp(-.15,.15)).clamp(limits[:,:,0],limits[:,:,1])
        return q
