"""Check open GARMI finger geometry along a GraspGen approach."""
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


@dataclass
class GraspClearance:
    object_mesh: trimesh.Trimesh
    robot_description: str
    finger_travel: float = .0398
    minimum_clearance: float = .002
    sampling_step: float = .001
    manager: object = field(init=False)
    parts: list = field(init=False)

    def __post_init__(self):
        self.manager = trimesh.collision.CollisionManager()
        self.manager.add_object('target', self.object_mesh)
        root = ET.fromstring(self.robot_description)
        self.parts = []
        for side, sign in [('left', 1), ('right', -1)]:
            finger = root.find(f"link[@name='right_fr3_{side}finger']")
            pieces = []
            for collision in finger.findall('collision'):
                origin = collision.find('origin')
                placement = np.eye(4)
                placement[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                placement[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                pieces.append(trimesh.creation.box(extents=np.fromstring(collision.find('geometry/box').get('size'), sep=' '), transform=placement))
            hand_finger = np.eye(4)
            hand_finger[:3, 3] = [0, sign*self.finger_travel, .0584]
            if sign < 0:
                hand_finger[:3, :3] = Rotation.from_euler('z', np.pi).as_matrix()
            mesh = trimesh.util.concatenate(pieces)
            mesh.apply_transform(hand_finger)
            self.parts.append(mesh)
        palm_path = root.find("link[@name='right_fr3_hand']/collision/geometry/mesh").get('filename')
        self.parts.append(trimesh.load(palm_path, force='mesh'))

    def check(self, mesh_to_gripper, approach_distance=.08):
        canonical_to_hand = np.eye(4)
        canonical_to_hand[:3, :3] = Rotation.from_euler('z', -np.pi/2).as_matrix()
        hand = mesh_to_gripper @ canonical_to_hand
        minimum = float('inf')
        # The half-step padding bounds translation between samples as well.
        required = self.minimum_clearance + self.sampling_step/2
        for offset in np.linspace(0., approach_distance, int(np.ceil(approach_distance/self.sampling_step))+1):
            placement = hand.copy()
            placement[:3, 3] -= mesh_to_gripper[:3, 2]*offset
            for mesh in self.parts:
                distance = float(self.manager.min_distance_single(mesh, transform=placement))
                minimum = min(minimum, distance)
                if distance < required:
                    return dict(passed=False, clearance_m=minimum, required_m=required,
                                offset_m=float(offset), finger_travel_m=self.finger_travel)
        return dict(passed=True, clearance_m=minimum, required_m=required,
                    finger_travel_m=self.finger_travel)
