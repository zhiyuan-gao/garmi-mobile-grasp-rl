import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cram_vrb_lab.robots.garmi.joints import load_patched_urdf
from garmi_cram_rl.gripper_geometry import pad_offset_from_tcp


def test_handle_is_centered_on_flat_pads_instead_of_the_sloped_collision_box():
    urdf = load_patched_urdf()
    robot = ET.fromstring(urdf)
    offset = pad_offset_from_tcp(urdf)
    finger = robot.find("link[@name='left_fr3_leftfinger']")
    pad = finger.findall('collision')[-1]
    center = np.fromstring(pad.find('origin').get('xyz'), sep=' ')
    size = np.fromstring(pad.find('geometry/box').get('size'), sep=' ')
    base = np.fromstring(robot.find("joint[@name='left_fr3_finger_joint1']/origin").get('xyz'), sep=' ')
    tcp = np.fromstring(robot.find("joint[@name='left_fr3_hand_tcp_joint']/origin").get('xyz'), sep=' ')
    assert tcp[2] + offset == pytest.approx(base[2] + center[2])
    assert tcp[2] < base[2] + center[2] - size[2]/2
    # The 11.8 mm authored handle fits wholly within the flat pad's axial span.
    handle_radius = .0059
    assert offset > 0
    assert handle_radius < size[2]/2


def test_invalid_pad_geometry_is_rejected():
    tree = ET.fromstring(load_patched_urdf())
    finger = tree.find("link[@name='left_fr3_leftfinger']")
    for collision in list(finger.findall('collision')):
        finger.remove(collision)
    with pytest.raises(ValueError, match='No flat opposing pad'):
        pad_offset_from_tcp(ET.tostring(tree, encoding='unicode'))
