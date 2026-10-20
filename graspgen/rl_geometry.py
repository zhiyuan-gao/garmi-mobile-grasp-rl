"""Static checks with GARMI's exported cooked robot collision geometry."""
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from trimesh.collision import CollisionManager

ROOT = Path(__file__).resolve().parents[1]


def origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3,3] = np.fromstring(node.get('xyz','0 0 0'),sep=' ')
        t[:3,:3] = Rotation.from_euler('xyz',np.fromstring(node.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t


class RobotCollision:
    def __init__(self, object_mesh, opening_m=.08):
        assert 0 < opening_m <= .08
        tree = ET.parse(ROOT/'rl/assets/garmi_training.urdf').getroot()
        bychild = {j.find('child').get('link'):j for j in tree.findall('joint')}
        parts = {}
        for record, mesh in pickle.load((ROOT/'assets/kitchen/cooked-model.pkl').open('rb')):
            if record['parent'] == 'Robot': parts.setdefault(record['body'],[]).append(mesh)
        self.meshes = {name:trimesh.util.concatenate(ms) for name,ms in parts.items()}
        self.chains = {}
        for name in parts:
            chain = []; child = name
            while child in bychild:
                joint = bychild[child]
                index = next((i for i in range(7) if joint.get('name') == f'right_fr3_joint{i+1}'),None)
                transform = origin(joint.find('origin'))
                if 'finger_joint' in joint.get('name'):
                    shift = np.eye(4)
                    shift[:3,3] = np.fromstring(joint.find('axis').get('xyz'),sep=' ')*(opening_m/2)
                    transform = transform@shift
                chain.append((transform,index))
                child = joint.find('parent').get('link')
            self.chains[name] = list(reversed(chain))
        groups = {'base_link':'base_link'}; pending = list(tree.findall('joint')); self.ignored=set()
        while pending:
            ready=[]
            for joint in pending:
                parent,child=joint.find('parent').get('link'),joint.find('child').get('link')
                if parent not in groups: continue
                separate = child in parts
                groups[child]=child if separate else groups[parent]
                if separate:self.ignored.add(frozenset((groups[parent],child)))
                ready.append(joint)
            if not ready: raise ValueError('Disconnected URDF')
            pending=[j for j in pending if j not in ready]
        for el in ET.parse(ROOT/'assets/garmi.srdf').getroot().findall('disable_self_collision'):
            a,b=el.get('link1'),el.get('link2')
            if a in parts and b in parts:self.ignored.add(frozenset((a,b)))
        self.ignored.add(frozenset(('right_fr3_leftfinger','right_fr3_rightfinger')))
        self.robot = CollisionManager()
        for name,mesh in self.meshes.items():self.robot.add_object(name,mesh)
        self.scene = CollisionManager()
        for i,(record,mesh) in enumerate(pickle.load((ROOT/'assets/kitchen/scene_meshes.pkl').open('rb'))):
            self.scene.add_object(str(i),mesh.convex_hull if record['type']=='Mesh' else mesh)
        self.target=CollisionManager();self.target.add_object('object',object_mesh)

    def check(self, base, joints, object_pose):
        for name,chain in self.chains.items():
            t=base.copy()
            for fixed,index in chain:
                t=t@fixed
                if index is not None:
                    c,s=np.cos(joints[index]),np.sin(joints[index])
                    r=np.eye(4);r[:2,:2]=[[c,-s],[s,c]];t=t@r
            self.robot.set_transform(name,t)
        if self.robot.in_collision_other(self.scene):return 'furniture_collision'
        hit,pairs=self.robot.in_collision_internal(return_names=True)
        if hit and any(frozenset(pair) not in self.ignored for pair in pairs):return 'self_collision'
        self.target.set_transform('object',object_pose)
        if self.robot.in_collision_other(self.target):return 'open_robot_object_collision'
        return None
