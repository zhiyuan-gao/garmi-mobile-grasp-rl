"""Calibrated RGB-D observations from a virtual camera at GARMI's head."""

from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
from scipy.spatial.transform import Rotation


def head_transform(urdf):
    joints = {
        j.find("child").get("link"): j
        for j in ET.parse(urdf).getroot().findall("joint")
    }
    chain = []
    child = "head"
    while child in joints:
        joint = joints[child]
        if joint.get("type") != "fixed":
            raise ValueError(
                "Head camera requires the frozen training torso/head chain"
            )
        origin = joint.find("origin")
        transform = np.eye(4)
        if origin is not None:
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
        chain.append(transform)
        child = joint.find("parent").get("link")
    result = np.eye(4)
    for transform in reversed(chain):
        result = result @ transform
    return result


class HeadCamera:
    def __init__(self, backend):
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.sensors.camera")
        from isaacsim.sensors.camera import Camera

        self.backend = backend
        self.mount = head_transform(
            backend.bundle.workspace / "rl/assets/garmi_training.urdf"
        )
        lens = np.eye(4)
        lens[:3, 3] = [0.13, 0.034, 0.02]
        # Same lens position as upstream GARMI, with a fixed 30-degree downward
        # optical tilt because the training robot has frozen head joints.
        lens[:3, :3] = (
            Rotation.from_euler("y", 30, degrees=True)
            * Rotation.from_quat([-0.5, 0.5, -0.5, 0.5])
        ).as_matrix()
        self.mount = self.mount @ lens
        self.camera = Camera(
            "/World/RoboKudoHeadCamera", resolution=(960, 540), frequency=20
        )
        self.camera.initialize()
        fov = getattr(backend, "perception_fov_degrees", None)
        if fov is not None:
            self.camera.set_focal_length(
                self.camera.get_horizontal_aperture() / (2 * np.tan(np.deg2rad(fov) / 2))
            )
        self.camera.set_clipping_range(0.08, 5.0)
        self.camera.add_distance_to_image_plane_to_frame()

    def capture(self, destination):
        robot = self.backend.robot.data
        root = np.eye(4)
        root[:3, 3] = robot.root_pos_w[0].cpu().numpy()
        root[:3, :3] = Rotation.from_quat(
            robot.root_quat_w[0].cpu().numpy()[[1, 2, 3, 0]]
        ).as_matrix()
        task_camera = root @ self.mount
        self.camera.set_world_pose(
            task_camera[:3, 3],
            Rotation.from_matrix(task_camera[:3, :3]).as_quat()[[3, 0, 1, 2]],
            camera_axes="ros",
        )
        for _ in range(12):
            self.backend.env.sim.render()
        rgb = np.asarray(self.camera.get_rgba())[..., :3].copy()
        depth = self.camera.get_depth(device="cpu")
        if depth is None or depth.shape != rgb.shape[:2]:
            raise RuntimeError("Head camera has not produced aligned RGB and depth")
        depth = np.asarray(depth, dtype=np.float32).copy()
        task_to_map = np.eye(4)
        task_to_map[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
        task_to_map[:3, 3] = [0.3, 6.35, 0.0]
        transform = task_to_map @ task_camera
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "rgbd.npz",
            rgb=rgb,
            depth_m=depth,
            intrinsics=self.camera.get_intrinsics_matrix().cpu().numpy(),
            map_T_camera=transform,
        )
        import imageio.v2 as imageio

        imageio.imwrite(destination / "rgb.png", rgb)
        (destination / "camera.json").write_text(
            json.dumps(
                dict(
                    map_T_camera=transform.tolist(),
                    base_T_camera=self.mount.tolist(),
                    resolution=[960, 540],
                    depth="optical_z_metres",
                    source="rendered_RGB_D",
                    simulation_seconds=self.backend.control_step
                    * self.backend.env.step_dt,
                ),
                indent=2,
            )
        )
        return destination / "rgbd.npz"


class ReviewCamera:
    """Fixed third-person recording camera, independent of policy perception."""

    def __init__(self, backend, path):
        from isaacsim.sensors.camera import Camera
        import imageio.v2 as imageio

        self.backend = backend
        self.camera = Camera(
            "/World/RoboKudoReviewCamera", resolution=(960, 540), frequency=10
        )
        self.camera.initialize()
        self.camera.set_focal_length(
            self.camera.get_horizontal_aperture() / (2 * np.tan(np.deg2rad(70) / 2))
        )
        eye = backend.frame.position_to_task([1.25, 5.9, 2.2])
        target = backend.frame.position_to_task([0.15, 6.85, 1.05])
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = Rotation.from_matrix(
            np.column_stack([right, down, forward])
        ).as_quat()[[3, 0, 1, 2]]
        self.camera.set_world_pose(eye, rotation, camera_axes="ros")
        self.writer = imageio.get_writer(path, fps=10, macro_block_size=2)

    def append(self):
        self.backend.env.sim.render()
        rgba = np.asarray(self.camera.get_rgba())
        if rgba.ndim == 3 and rgba.shape[0] > 0:
            self.writer.append_data(rgba[:, :, :3])

    def close(self):
        self.writer.close()
