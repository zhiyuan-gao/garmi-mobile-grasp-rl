"""Use the existing CRAM learned pick-up action with a mug-specific world body."""

import time
import json
from pathlib import Path
import trimesh
from cram_demo import GraspDemo
from mobile_grasp import (
    BodySpecification,
    Connection6DoFSpecification,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Mug


class MugDemo(GraspDemo):
    skill_object_name = "mug"

    def note_pose_update(self, message):
        if "robokudo_mug" in json.loads(message.data):
            self.pose_received_at = time.monotonic()

    def setup_object(self):
        state = self.rl.snapshot()["snapshot"]
        name = "robokudo_mug"
        self.mesh_path = Path(
            "/workspace/household/mug/assets/ycb/025_mug/surface-v3/repaired.obj"
        )
        self.mesh = trimesh.load(self.mesh_path, force="mesh")
        initial = HomogeneousTransformationMatrix.from_xyz_rpy(
            *state["object_map_position"], *state["object_map_euler"]
        )
        if self.world.is_kinematic_structure_entity_in_world_by_name(name):
            raise RuntimeError("Use a fresh planning world for this isolated mug test")
        Mug.get_annotation_specification(
            name,
            BodySpecification.mesh(name, str(self.mesh_path), parent_T_self=initial),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(self.world)
        self.body = self.world.get_body_by_name(name)
        self.annotation = next(
            a
            for a in self.world.get_semantic_annotations_by_type(Mug)
            if a.root is self.body
        )
        self.sync._synced.append(name)
        deadline = time.monotonic() + 5
        while name not in self.sync.tracked_poses():
            if time.monotonic() > deadline:
                raise TimeoutError("Perceived mug pose was not published")
            time.sleep(0.05)
        self.sync.pull(self.world)
        self.record(
            "settled",
            object=self.object_pose(),
            backend="robokudo_mug",
            mesh=str(self.mesh_path),
        )
