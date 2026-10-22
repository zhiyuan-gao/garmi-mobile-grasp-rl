"""Scripted camera aims must remain valid pivots for interactive navigation."""
import math

import pytest

pytest.importorskip('pxr')
from pxr import Gf, Sdf, Usd, UsdGeom

from garmi_cram_rl.viewport import set_navigation_target


@pytest.mark.parametrize('session_override', [False, True])
@pytest.mark.parametrize('eye,target', [
    ((-1.4, -.9, 2.5), (.65, -.1, 1.1)),
    ((-.8, 3.6, 2.6), (1.2, 5.35, .95)),
])
def test_camera_pivot_is_local_and_reconstructs_world_target(eye, target, session_override):
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, '/Camera')
    transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target),
                                      Gf.Vec3d(0, 0, 1)).GetInverse()
    camera.AddTransformOp().Set(transform)
    camera.GetPrim().CreateAttribute('omni:kit:centerOfInterest',
        Sdf.ValueTypeNames.Vector3d, custom=True).Set(Gf.Vec3d(*target))
    if session_override:
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            camera.GetPrim().GetAttribute('omni:kit:centerOfInterest').Set(Gf.Vec3d(0, 0, -866.0254))
    edit_target = stage.GetEditTarget().GetLayer()

    set_navigation_target(stage, '/Camera', target)

    pivot = camera.GetPrim().GetAttribute('omni:kit:centerOfInterest').Get()
    assert tuple(pivot) == pytest.approx((0., 0., -math.dist(eye, target)))
    assert tuple(transform.Transform(pivot)) == pytest.approx(target)
    assert UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) == transform
    assert stage.GetEditTarget().GetLayer() == edit_target
