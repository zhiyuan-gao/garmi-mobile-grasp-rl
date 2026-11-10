"""A perception-fed bowl backend with a separate physical scoring channel."""

import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from garmi_cram_rl.lab_backend import LabBackend
from garmi_cram_rl.contract import MapFrame
from camera import HeadCamera


class ObjectDataView:
    """Only the estimated pose is exposed to policy observation construction."""

    def __init__(self, backend):
        self.backend = backend

    @property
    def root_pos_w(self):
        return self.backend.estimated_position

    @property
    def root_quat_w(self):
        return self.backend.estimated_rotation


class SceneView:
    def __init__(self, backend):
        self.backend = backend

    def __getitem__(self, name):
        if name == "object":
            return self.backend.observed_object
        return self.backend.env.scene[name]

    def __getattr__(self, name):
        return getattr(self.backend.env.scene, name)


class EnvironmentView:
    def __init__(self, backend):
        self.backend = backend
        self.scene = SceneView(backend)

    def __getattr__(self, name):
        return getattr(self.backend.env, name)


class ObservedMDP:
    def __init__(self, original, observed):
        self.original, self.observed = original, observed

    def observation(self, env):
        return self.original.observation(self.observed)

    def __getattr__(self, name):
        return getattr(self.original, name)


class PerceivedBowlBackend(LabBackend):
    def __init__(self, bundle, base_map, bowl_map, output, capture_only=False):
        self.output = Path(output)
        self.estimate = None
        super().__init__(
            bundle,
            base_map,
            bowl_map,
            render=False,
            full_scene=True,
            configure=lambda cfg: setattr(cfg, "ui_window_class_type", None),
        )
        # Settle the real bowl, without using its resulting pose in the detector.
        for _ in range(30):
            self.step_classical(time.monotonic())
        self.sensor = HeadCamera(self)
        frame = self.sensor.capture(self.output / "perception")
        if capture_only:
            return
        command = [
            "bash",
            "-c",
            "source /opt/ros/jazzy/setup.bash; source /home/jovyan/cram-vrb-lab/ros2_ws/install/setup.bash; "
            'export PYTHONPATH=/workspace/cram-vrb-lab:$PYTHONPATH; export OMP_NUM_THREADS=2; exec "$@"',
            "robokudo",
            "/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python",
            str(Path(__file__).with_name("detect.py")),
            str(frame),
        ]
        with (self.output / "perception/detector.log").open("w") as log:
            subprocess.run(
                command,
                check=True,
                timeout=90,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    k: v
                    for k, v in __import__("os").environ.items()
                    if k
                    not in (
                        "PYTHONHOME",
                        "PYTHONPATH",
                        "LD_LIBRARY_PATH",
                        "ISAAC_PATH",
                        "CARB_APP_PATH",
                        "EXP_PATH",
                        "CONDA_PREFIX",
                        "PYTHONEXE",
                    )
                },
            )
        self.estimate = json.loads((frame.parent / "estimate.json").read_text())
        pos = self.frame.position_to_task(self.estimate["position_map"])
        quat = self.frame.quaternion_to_task(self.estimate["quaternion_wxyz"])
        self.estimated_position = torch.tensor(
            [pos.tolist()], device=self.env.device, dtype=torch.float32
        )
        self.estimated_rotation = torch.tensor(
            [quat.tolist()], device=self.env.device, dtype=torch.float32
        )
        from types import SimpleNamespace

        self.observed_object = SimpleNamespace(data=ObjectDataView(self))
        self.observed_env = EnvironmentView(self)
        self.original_mdp = self.mdp
        self.mdp = ObservedMDP(self.mdp, self.observed_env)
        self.env.fixed_finish.env = self.observed_env
        self.audit = (self.output / "policy-inputs.jsonl").open("w", buffering=1)
        self.held_offset = None
        self.verify_input_isolation()
        # Truth is recorded for evaluation only; the detector has already exited.
        truth = super().snapshot()
        error = np.array(self.estimate["position_map"]) - truth["object_map_position"]
        (self.output / "localization-score.json").write_text(
            json.dumps(
                dict(
                    estimated=self.estimate,
                    truth_position_map=truth["object_map_position"],
                    error_m=error.tolist(),
                    position_error_m=float(np.linalg.norm(error)),
                ),
                indent=2,
            )
        )

    def verify_input_isolation(self):
        baseline = self.mdp.observation(self.env).clone()
        objects = self.env.scene.rigid_objects
        physical_object = objects["object"]
        try:
            # Fail immediately if policy observation construction reaches the
            # simulator's object rather than the explicit perception view.
            objects["object"] = object()
            separated = self.mdp.observation(self.env)
            self.original_mdp.geometry(self.env.fixed_finish.env)
            assert torch.equal(baseline, separated)
        finally:
            objects["object"] = physical_object
        original = self.estimated_position.clone()
        try:
            self.estimated_position[:, 0] += 0.02
            shifted = self.mdp.observation(self.env)
            difference = float(torch.max(torch.abs(shifted - baseline)))
            assert difference > 0.01
        finally:
            self.estimated_position[:] = original
        (self.output / "input-isolation.json").write_text(
            json.dumps(
                dict(
                    policy_works_without_simulated_object=True,
                    fixed_finish_uses_perceived_geometry=True,
                    observation_changes_with_perceived_pose=True,
                    maximum_observation_change=difference,
                    physical_scoring_uses_simulator=True,
                ),
                indent=2,
            )
        )

    def snapshot(self):
        state = super().snapshot()
        if self.estimate is None:
            raise RuntimeError("RoboKudo pose is not ready")
        object_q = self.frame.quaternion_to_map(
            self.estimated_rotation[0].cpu().numpy()
        )
        state.update(
            object_map_position=self.frame.position_to_map(
                self.estimated_position[0].cpu().numpy()
            ).tolist(),
            object_map_quaternion_wxyz=object_q.tolist(),
            object_map_euler=Rotation.from_quat(object_q[[1, 2, 3, 0]])
            .as_euler("xyz")
            .tolist(),
            object_velocity=[0.0] * 6,
            perception=dict(
                source=self.estimate["position_source"],
                mode="single_frame_static_until_lift",
                surface_fit_rmse_m=self.estimate["surface_fit_rmse_m"],
            ),
        )
        return state

    def step_policy(self):
        # Once closing ends, carry the last estimate with the measured gripper.
        # This is kinematic tracking, not a new measurement of the object.
        phase = int(self.env.fixed_finish.phase[0])
        if phase >= 3:
            from isaaclab.utils.math import (
                subtract_frame_transforms,
                combine_frame_transforms,
            )

            hp, hq, _, _ = self.original_mdp.geometry(self.observed_env)
            if self.held_offset is None:
                self.held_offset = subtract_frame_transforms(
                    hp, hq, self.estimated_position, self.estimated_rotation
                )
            self.estimated_position[:], self.estimated_rotation[:] = (
                combine_frame_transforms(hp, hq, *self.held_offset)
            )
        observation = self.mdp.observation(self.env)
        hp, hq, gp, gq = self.original_mdp.geometry(self.observed_env)
        self.audit.write(
            json.dumps(
                dict(
                    step=self.control_step,
                    phase=phase,
                    source="robokudo",
                    object_position_task=self.estimated_position[0].cpu().tolist(),
                    grasp_position_task=gp[0].cpu().tolist(),
                    observation=observation[0].cpu().tolist(),
                )
            )
            + "\n"
        )
        return super().step_policy()
