"""Locate the opposing flat finger pads in the authored GARMI collision model."""
import xml.etree.ElementTree as ET

import numpy as np


def pad_offset_from_tcp(urdf, side='left'):
    robot = ET.fromstring(urdf)
    finger = robot.find(f"link[@name='{side}_fr3_leftfinger']")
    candidates = []
    for collision in finger.findall('collision'):
        box = collision.find('geometry/box')
        if box is None:
            continue
        origin = collision.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        size = np.fromstring(box.get('size'), sep=' ')
        if np.max(np.abs(rpy)) < 1.e-6 and abs(xyz[1] - size[1]/2) < .0002:
            candidates.append(xyz)
    if not candidates:
        raise ValueError('No flat opposing pad found in the finger collision model')
    pad = max(candidates, key=lambda p: p[2])
    finger_origin = robot.find(f"joint[@name='{side}_fr3_finger_joint1']/origin")
    tcp_origin = robot.find(f"joint[@name='{side}_fr3_hand_tcp_joint']/origin")
    hand_pad_z = np.fromstring(finger_origin.get('xyz'), sep=' ')[2] + pad[2]
    hand_tcp_z = np.fromstring(tcp_origin.get('xyz'), sep=' ')[2]
    return float(hand_pad_z - hand_tcp_z)
