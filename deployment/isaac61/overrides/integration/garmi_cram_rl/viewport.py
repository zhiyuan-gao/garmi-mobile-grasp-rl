"""Camera navigation for the streamed Isaac Sim desktop."""


def configure_streamed_navigation():
    """Apply navigation speeds to the live model after Kit loads preferences."""
    from omni.kit.viewport.utility import get_active_viewport_window

    window = get_active_viewport_window()
    item = window._find_viewport_layer('Camera', 'manipulator') if window else None
    if item is None:
        raise RuntimeError('The streamed viewport camera manipulator is unavailable')
    # Persistent user preferences can override the .kit startup defaults.
    model = item.layer.manipulator.model
    model.set_floats('move_speed', [1.] * 3)
    model.set_floats('tumble_speed', [1.])
    model.set_floats('look_speed', [1., .5, 0.])
    model.set_floats('scroll_speed', [.025] * 3)


def set_navigation_target(stage, camera_path, target):
    """Keep Kit's orbit/zoom pivot in camera-local coordinates after aiming it."""
    from pxr import Gf, Sdf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(camera_path)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    local_target = transform.GetInverse().Transform(Gf.Vec3d(*map(float, target)))
    # ViewportManager.set_camera_view in Sim 6.1 writes a world-space target to
    # this attribute, while Kit's camera manipulator reads a camera-local pivot.
    # Kit's implicit camera already has this attribute in the session layer.
    # Writing into the root layer is silently shadowed by that stronger opinion
    # (initially a pivot hundreds of metres away), amplifying pan and zoom.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        prim.CreateAttribute('omni:kit:centerOfInterest', Sdf.ValueTypeNames.Vector3d,
                             custom=True).Set(local_target)
