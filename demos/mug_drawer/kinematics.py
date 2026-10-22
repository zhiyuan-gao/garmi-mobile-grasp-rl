"""Object-independent geometric action adapters, in metres and radians."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def transform(position, rotation):
    t = np.eye(4); t[:3, :3] = rotation; t[:3, 3] = position
    return t


def pose_error(target, current):
    return np.r_[target[:3, 3]-current[:3, 3],
                 Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()]


class GarmiKinematics:
    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        bychild = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain = []; name = 'right_fr3_hand'
        while name in bychild:
            j = bychild[name]; chain.append(j); name = j.find('parent').get('link')
        self.chain = []; self.limits = np.zeros((7, 2))
        for j in reversed(chain):
            o = j.find('origin')
            xyz = np.fromstring(o.get('xyz', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
            rpy = np.fromstring(o.get('rpy', '0 0 0'), sep=' ') if o is not None else np.zeros(3)
            index = next((i for i in range(7) if j.get('name') == f'right_fr3_joint{i+1}'), None)
            axis = np.fromstring(j.find('axis').get('xyz'), sep=' ') if index is not None else np.zeros(3)
            if index is not None:
                lim = j.find('limit'); self.limits[index] = [float(lim.get('lower')), float(lim.get('upper'))]
            self.chain.append((transform(xyz, Rotation.from_euler('xyz', rpy).as_matrix()), index, axis))

    def forward(self, q):
        t = np.eye(4)
        for origin, i, axis in self.chain:
            t = t @ origin
            if i is not None: t = t @ transform(np.zeros(3), Rotation.from_rotvec(axis*q[i]).as_matrix())
        return t

    def solve(self, target, initial, max_nfev=80):
        initial = np.clip(initial, self.limits[:, 0]+1e-6, self.limits[:, 1]-1e-6)
        fit = least_squares(lambda q: np.r_[pose_error(target, self.forward(q)), .002*(q-initial)],
                            initial, bounds=(self.limits[:, 0]+1e-6, self.limits[:, 1]-1e-6),
                            max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        return fit.x, pose_error(target, self.forward(fit.x))
