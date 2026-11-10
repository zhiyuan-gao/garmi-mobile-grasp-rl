"""Model-origin and visibility checks independent of Isaac rendering."""

import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "bowl_detector", Path(__file__).parents[1] / "detect.py"
)
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)


def test_surface_alignment_recovers_offset_asset_origin():
    bounds = np.array([[-0.06949, -0.07179, -0.03247], [0.07017, 0.06721, 0.03465]])
    x = np.linspace(bounds[0, 0], bounds[1, 0], 70)
    y = np.linspace(bounds[0, 1], bounds[1, 1], 70)
    z = np.linspace(bounds[0, 2], bounds[1, 2], 30)
    xx, yy = np.meshgrid(x, y)
    top = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, bounds[1, 2])]
    xx, zz = np.meshgrid(x, z)
    front = np.c_[xx.ravel(), np.full(xx.size, bounds[0, 1]), zz.ravel()]
    yy, zz = np.meshgrid(y, z)
    left = np.c_[np.full(yy.size, bounds[0, 0]), yy.ravel(), zz.ravel()]
    model = np.r_[top, front, left]
    origin = np.array([0.2, 7.4, 0.945 - bounds[0, 2]])
    result = detector.estimate_bowl(model[::2] + origin, model, bounds, 0.945)
    np.testing.assert_allclose(result["position_map"], origin, atol=0.0002)


def test_incomplete_cluster_is_rejected():
    bounds = [[-0.07, -0.07, -0.03], [0.07, 0.07, 0.035]]
    points = np.random.default_rng(0).uniform(
        [-0.01, -0.01, 0.99], [0.01, 0.01, 1.01], (100, 3)
    )
    with pytest.raises(ValueError, match="extent"):
        detector.estimate_bowl(points, points, bounds, 0.945)
