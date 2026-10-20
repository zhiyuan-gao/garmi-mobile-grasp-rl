"""Conservative solid collision checks for fixed-reference kitchen goals."""
import itertools
import json
import pickle
from pathlib import Path

import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFLIGHT = ROOT / 'assets/kitchen'
BANK = ROOT / 'assets/cup/bank.json'


def convex_geometry(mesh):
    hull = mesh.convex_hull
    faces = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).astype(np.int32).ravel()
    return fcl.Convex(np.asarray(hull.vertices), len(hull.faces), faces)


class GoalGeometry:
    def __init__(self, bank_path=BANK):
        self.bank = json.loads(Path(bank_path).read_text())['objects'][0]
        self.scene_meshes = pickle.load((PREFLIGHT / 'scene_meshes.pkl').open('rb'))
        self.scene_geometries = []
        self.scene_objects = []
        self.scene_approximations = []
        for record, original in self.scene_meshes:
            mesh = original.copy()
            if record['type'] == 'Mesh':
                # These are distant room walls, windows and a chair. Solid
                # enclosing boxes also handle zero-thickness wall triangles.
                bounds = mesh.bounds
                extent = np.maximum(bounds[1] - bounds[0], 2.e-6)
                mesh = trimesh.creation.box(extent)
                mesh.apply_translation(bounds.mean(axis=0))
                self.scene_approximations.append(record['path'])
            elif record['type'] == 'Cylinder':
                # Export uses an inscribed 48-gon: enlarge to enclose the
                # analytic cylinder (also slightly lengthens its axis).
                center = mesh.vertices.mean(axis=0)
                mesh.vertices = center + (mesh.vertices - center) / np.cos(np.pi / 48)
            geom = convex_geometry(mesh)
            self.scene_geometries.append(geom)
            self.scene_objects.append(fcl.CollisionObject(geom))
        self.scene = fcl.DynamicAABBTreeCollisionManager()
        self.scene.registerObjects(self.scene_objects)
        self.scene.setup()

        link7_T_hand = np.array(json.loads((ROOT / 'rl/assets/manifest.json').read_text())['link7_T_hand'])
        parts = []
        for record, mesh in pickle.load((PREFLIGHT / 'cooked-model.pkl').open('rb')):
            body = record['body']
            if body not in ('right_fr3_link7', 'right_fr3_leftfinger', 'right_fr3_rightfinger'):
                continue
            hand_T_body = np.eye(4)
            if body == 'right_fr3_link7':
                hand_T_body = np.linalg.inv(link7_T_hand)
            else:
                sign = 1 if body.endswith('leftfinger') else -1
                hand_T_body[:3, 3] = [0, sign * .04, .0584]
                if sign == -1:
                    hand_T_body[:3, :3] = Rotation.from_euler('z', np.pi).as_matrix()
            part = mesh.copy()
            part.apply_transform(hand_T_body)
            parts.append(part)
        self.parts = parts
        object_pose = np.eye(4)
        object_pose[:3, :3] = Rotation.from_euler('z', -90, degrees=True).as_matrix()
        object_pose[2, 3] = .946 - self.bank['bounds'][0][2]
        tool = np.eye(4)
        tool[:3, :3] = Rotation.from_euler('z', -90, degrees=True).as_matrix()
        self.reference_meshes = []
        for ref in self.bank['references']:
            relative_hand = np.array(ref['mesh_T_gripper']) @ tool
            runtime_hand = np.eye(4)
            runtime_hand[:3, 3] = ref['position']
            runtime_hand[:3, :3] = Rotation.from_quat(np.array(ref['quaternion'])[[1, 2, 3, 0]]).as_matrix()
            if not np.allclose(relative_hand, runtime_hand, atol=1.e-6, rtol=0.):
                raise ValueError(f"Collision pose differs from policy reference: {ref['id']}")
            hand = object_pose @ relative_hand
            meshes = []
            for part in parts:
                mesh = part.copy()
                mesh.apply_transform(hand)
                meshes.append(mesh)
            self.reference_meshes.append(meshes)

    def query(self, reference_index, half_cell_xy=(0., 0.), margin=0.):
        # Map X is -task Y; map Y is task X. Fixed yaw means only translation
        # varies over the cell. Convex hull of all corner translations is the
        # exact swept volume of each convex part, with an extra margin cube.
        radius = np.array([half_cell_xy[1] + margin, half_cell_xy[0] + margin, margin])
        offsets = np.array(list(itertools.product(*[(-v, v) for v in radius])))
        meshes = []
        for mesh in self.reference_meshes[reference_index]:
            points = (mesh.vertices[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
            meshes.append(trimesh.convex.convex_hull(points))
        return HandQuery(meshes, self.scene)


class HandQuery:
    def __init__(self, meshes, scene):
        self.geometries = [convex_geometry(mesh) for mesh in meshes]
        self.objects = [fcl.CollisionObject(g) for g in self.geometries]
        self.manager = fcl.DynamicAABBTreeCollisionManager()
        self.manager.registerObjects(self.objects)
        self.manager.setup()
        self.scene = scene

    def collides(self, map_xy):
        translation = np.array([map_xy[1] - 6.35, .3 - map_xy[0], 0.])
        transform = fcl.Transform(np.eye(3), translation)
        for obj in self.objects:
            obj.setTransform(transform)
        self.manager.update()
        data = fcl.CollisionData()
        self.manager.collide(self.scene, data, fcl.defaultCollisionCallback)
        return bool(data.result.is_collision)
