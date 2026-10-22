"""Map frames, frozen resources and admission checks for a grasp request."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workspace_path(path, root):
    value = str(path)
    return Path(root) / value.removeprefix('/workspace/') if value.startswith('/workspace/') else Path(value)


@dataclass(frozen=True)
class MapFrame:
    """The existing kitchen transform: map_T_task = (0.3, 6.35, yaw=90°)."""

    def position_to_map(self, position):
        x, y, z = position
        return np.array([.3 - y, 6.35 + x, z])

    def position_to_task(self, position):
        x, y, z = position
        return np.array([y - 6.35, .3 - x, z])

    def quaternion_to_map(self, quaternion_wxyz):
        q = np.asarray(quaternion_wxyz)
        rotation = Rotation.from_euler('z', np.pi / 2) * Rotation.from_quat(q[[1, 2, 3, 0]])
        return rotation.as_quat()[[3, 0, 1, 2]]

    def quaternion_to_task(self, quaternion_wxyz):
        q = np.asarray(quaternion_wxyz)
        rotation = Rotation.from_euler('z', -np.pi / 2) * Rotation.from_quat(q[[1, 2, 3, 0]])
        return rotation.as_quat()[[3, 0, 1, 2]]


@dataclass
class Bundle:
    path: Path
    workspace: Path

    def __post_init__(self):
        self.path = Path(self.path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest['action_contract'] != 'base_velocity_joint_delta_v1':
            raise ValueError('This bridge requires the trained joint-increment policy')
        for name, expected in self.manifest['bundle_files'].items():
            if file_hash(self.path / name) != expected:
                raise ValueError(f'Bundle file changed: {name}')
        for name, expected in self.manifest['workspace_files'].items():
            if file_hash(workspace_path(name, self.workspace)) != expected:
                raise ValueError(f'Training resource changed: {name}')


def validate_start(snapshot, task, mask, reference=None):
    """Accept a parked, stationary robot and select from the original spatial mask.

    Tolerances here handle measured settling; they do not widen the training region.
    The bowl mask remains restricted to yaw zero, as in training.
    """
    base = np.asarray(snapshot['base_map_position'], dtype=float)
    bowl = np.asarray(snapshot['object_map_position'], dtype=float)
    arm = np.asarray(snapshot['arm_position'], dtype=float)
    park = np.array([0., -.785398, 0., -2.356194, 0., 1.570796, .785398])
    values = np.r_[base, bowl, arm, snapshot['base_velocity'], snapshot['joint_velocity'],
                   snapshot['base_map_yaw'], snapshot['object_map_euler']]
    if not np.isfinite(values).all():
        raise ValueError('Non-finite measured robot or object state')
    lower, upper = np.asarray(task['kitchen']['base_bounds_map_xy'])
    if ((base[:2] < lower) | (base[:2] > upper)).any():
        raise ValueError('Navigate into the trained counter-front region first')
    angle = np.arctan2(np.sin(snapshot['base_map_yaw'] - np.pi / 2),
                       np.cos(snapshot['base_map_yaw'] - np.pi / 2))
    if abs(angle) > .01:
        raise ValueError('Base must face the counter (90 degrees, tolerance 0.01 rad)')
    if np.max(np.abs(arm - park)) > .02:
        raise ValueError('Right arm must start in the training park posture')
    if 'joint_settling_range' in snapshot:
        # PhysX reports a repeatable ~0.04 rad/s substep velocity in the parked
        # arm despite <3e-6 rad position variation over two seconds. Require
        # measured position stability over 0.5 s, retaining a velocity ceiling.
        arm_settled = (snapshot['settling_window_seconds'] >= .45
                       and np.isfinite(snapshot['joint_settling_range']).all()
                       and np.max(snapshot['joint_settling_range']) < .002
                       and np.max(np.abs(snapshot['joint_velocity'])) < .1)
    else:
        arm_settled = np.max(np.abs(snapshot['joint_velocity'])) <= .02
    if np.max(np.abs(snapshot['base_velocity'])) > .01 or not arm_settled:
        raise ValueError('Wait for measured base and arm motion to settle')
    if np.max(np.abs(snapshot['object_map_euler'])) > 1.e-4:
        raise ValueError('Bowl must be upright at the yaw used by the validated mask')
    if abs(bowl[2] - mask.data['bowl_origin_z_m']) > .003:
        raise ValueError('Bowl is not resting at the training counter height')
    allowed = mask.allowed([bowl[:2].tolist()], 0.)[0].cpu().numpy()
    names = mask.data['reference_ids']
    candidates = [name for name, valid in zip(names, allowed) if valid]
    if reference is not None and reference not in candidates:
        raise ValueError('Requested reference is blocked or outside the trained bank')
    # Fixed ordering is reproducible; this is not a learned grasp-ranking rule.
    return reference if reference is not None else candidates[0]
