"""Geometry/frame regressions for the independent YCB reference selector."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('select_references',
    Path(__file__).resolve().parents[1]/'graspgen/select_references.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def test_conversion_matches_all_imported_bowl_references():
    selector.verify_bowl_conversion()


def test_swapping_parallel_fingers_is_not_a_new_grasp():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, :3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    poses[2, :3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    centers, rotations = selector.pose_components(poses)
    distance = selector.distances_to(centers, rotations, 0)
    assert distance[1] < 1e-6
    assert np.isclose(distance[2], 3.)


def test_distance_uses_contact_center_not_wrist_position():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, :3] = Rotation.from_euler('y', 30, degrees=True).as_matrix()
    tcp = np.array(selector.SETTINGS['contact_center_hand_m'])
    poses[1, :3, 3] = tcp - poses[1, :3, :3] @ tcp
    centers, rotations = selector.pose_components(poses)
    assert np.allclose(centers[0], centers[1])
    assert np.isclose(selector.distances_to(centers, rotations, 0)[1], 1.)


def test_farthest_selection_covers_sparse_region_instead_of_score_cluster():
    poses = np.repeat(np.eye(4)[None], 102, axis=0)
    poses[:100, 0, 3] = np.linspace(0, .01, 100)
    poses[100:, 0, 3] = [.2, -.2]
    scores = np.r_[np.linspace(.99, .9, 100), .6, .6]
    selected = selector.farthest_selection(poses, scores, count=3)
    assert set(selected) == {0, 100, 101}
    assert selector.coverage(poses, selected)['nearest_reference_distance_max'] < .34


def test_metric_is_invariant_to_common_object_frame_transform():
    rng = np.random.default_rng(55)
    poses = np.repeat(np.eye(4)[None], 12, axis=0)
    poses[:, :3, :3] = Rotation.random(12, random_state=rng).as_matrix()
    poses[:, :3, 3] = rng.normal(size=(12, 3))*.05
    common = np.eye(4)
    common[:3, :3] = Rotation.from_euler('xyz', [.4, -.6, .3]).as_matrix()
    common[:3, 3] = [.4, .8, -.3]
    d1 = selector.distances_to(*selector.pose_components(poses), 0)
    d2 = selector.distances_to(*selector.pose_components(common[None] @ poses), 0)
    assert np.allclose(d1, d2, atol=1e-6)
