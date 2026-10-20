"""Compare the original CRAM URDF with the frozen training robot on CPU."""
from pathlib import Path
import json
import sys
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cram-vrb-lab'))
from cram_vrb_lab.robots.garmi.joints import load_patched_urdf


def hand_pose(tree, positions, endpoint='right_fr3_hand'):
    by_child = {joint.find('child').get('link'): joint for joint in tree.findall('joint')}
    chain, name = [], endpoint
    while name in by_child:
        joint = by_child[name]
        chain.append(joint)
        name = joint.find('parent').get('link')
    result = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find('origin')
        transform = np.eye(4)
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        result = result @ transform
        displacement = np.eye(4)
        value = positions.get(joint.get('name'), 0.)
        if joint.get('type') != 'fixed':
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            if joint.get('type') == 'prismatic':
                displacement[:3, 3] = axis * value
            else:
                displacement[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
        result = result @ displacement
    return result


def test_cram_and_training_hand_frames_agree_with_frozen_lifts():
    original = ElementTree.fromstring(load_patched_urdf())
    training = ElementTree.parse(ROOT / 'rl/assets/garmi_training.urdf').getroot()
    fixed = json.loads((ROOT / 'rl/assets/manifest.json').read_text())['frozen_joint_positions']
    rng = np.random.default_rng(42)
    lower = [-2.75, -1.74, -2.75, -3., -2.73, .55, -2.89]
    upper = [2.75, 1.74, 2.75, -.19, 2.73, 4.51, 2.89]
    for joints in rng.uniform(lower, upper, (32, 7)):
        positions = dict(fixed, **{f'right_fr3_joint{i+1}': value for i, value in enumerate(joints)})
        np.testing.assert_allclose(hand_pose(original, positions), hand_pose(training, positions), atol=1.e-8)


def test_dual_arm_asset_preserves_both_hand_frames_and_original_left_joint_limits():
    if not (ROOT / 'integration/assets/drawer/garmi_dual.urdf').exists():
        pytest.skip('Build the optional drawer demo assets first')
    original = ElementTree.fromstring(load_patched_urdf())
    dual = ElementTree.parse(ROOT / 'integration/assets/drawer/garmi_dual.urdf').getroot()
    training = ElementTree.parse(ROOT / 'rl/assets/garmi_training.urdf').getroot()
    fixed = json.loads((ROOT / 'rl/assets/manifest.json').read_text())['frozen_joint_positions']
    for name in [f'left_fr3_joint{i}' for i in range(1, 8)] + [
            'left_fr3_finger_joint1', 'left_fr3_finger_joint2']:
        actual = dual.find(f"joint[@name='{name}']")
        expected = original.find(f"joint[@name='{name}']")
        assert actual.get('type') == expected.get('type')
        assert actual.find('limit').attrib == expected.find('limit').attrib
    rng = np.random.default_rng(7)
    lower = [-2.75, -1.74, -2.75, -3., -2.73, .55, -2.89]
    upper = [2.75, 1.74, 2.75, -.19, 2.73, 4.51, 2.89]
    for right, left in rng.uniform(lower, upper, (32, 2, 7)):
        positions = dict(fixed, **{f'right_fr3_joint{i+1}': q for i, q in enumerate(right)},
                        **{f'left_fr3_joint{i+1}': q for i, q in enumerate(left)})
        np.testing.assert_allclose(hand_pose(dual, positions), hand_pose(training, positions), atol=1.e-8)
        np.testing.assert_allclose(hand_pose(dual, positions, 'left_fr3_hand'),
                                   hand_pose(original, positions, 'left_fr3_hand'), atol=1.e-8)
