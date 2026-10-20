"""Convert cached GraspGen Panda poses into object-relative CRAM grasp frames."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class GraspAdapter:
    """Calibrate the canonical Panda opening axis to the GARMI hand and tool."""

    hand_to_tool: np.ndarray
    grasp_to_tool_rotation: np.ndarray

    def to_tool(self, mesh_to_gripper, body_to_mesh):
        # GraspGen's fingers open on X; GARMI's URDF fingers open on hand Y.
        gripper_to_hand = np.eye(4)
        gripper_to_hand[:3, :3] = Rotation.from_euler('z', -np.pi / 2).as_matrix()
        return body_to_mesh @ mesh_to_gripper @ gripper_to_hand @ self.hand_to_tool

    def to_grasp(self, mesh_to_gripper, body_to_mesh):
        result = self.to_tool(mesh_to_gripper, body_to_mesh)
        result[:3, :3] = result[:3, :3] @ self.grasp_to_tool_rotation.T
        return result


def load_candidates(cache_path, mesh_path):
    """Reject stale geometry, invalid poses and non-finite model scores."""
    data = np.load(cache_path, allow_pickle=False)
    expected = hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest()
    if str(data['mesh_sha256']) != expected:
        raise ValueError('Candidate mesh checksum does not match the simulation mesh')
    grasps, scores = data['grasps'], data['scores']
    if grasps.shape != (len(scores), 4, 4) or not len(scores):
        raise ValueError('Invalid candidate dimensions')
    if not np.isfinite(grasps).all() or not np.isfinite(scores).all():
        raise ValueError('Non-finite candidate')
    rotations = grasps[:, :3, :3]
    if not (np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
            and np.allclose(np.linalg.det(rotations), 1, atol=1e-4)):
        raise ValueError('Invalid candidate rotation')
    if not np.allclose(grasps[:, 3, :], [0, 0, 0, 1]):
        raise ValueError('Invalid homogeneous transform')
    return grasps, scores
