"""Run RoboKudo on RGB-D and localize a known object at its fixed upright orientation."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def estimate_bowl(points, model_points, bounds, plane_height):
    """Align the known upright object surface to its visible RGB-D cluster."""
    from scipy.spatial import cKDTree

    low, high = np.asarray(bounds)
    if len(points) < 80:
        raise ValueError("Too few visible object points")
    extent = np.ptp(points, axis=0)
    model_extent = high - low
    if np.any(extent[:2] < 0.6 * model_extent[:2]) or np.any(
        extent[:2] > 1.2 * model_extent[:2]
    ):
        raise ValueError("Visible extent does not match the known object")
    if abs(np.quantile(points[:, 2], 0.99) - (plane_height + model_extent[2])) > 0.012:
        raise ValueError("Cluster height does not match the object model")
    z = float(plane_height - low[2])
    initial = (np.min(points[:, :2], axis=0) + np.max(points[:, :2], axis=0)) / 2 - (
        low[:2] + high[:2]
    ) / 2
    tree = cKDTree(model_points)

    def residual(xy):
        local = points - np.r_[xy, z]
        _, indices = tree.query(local)
        return (local - model_points[indices]).ravel()

    fit = least_squares(
        residual,
        initial,
        bounds=(initial - 0.04, initial + 0.04),
        loss="soft_l1",
        f_scale=0.002,
        max_nfev=80,
        diff_step=1e-5,
        xtol=1e-9,
    )
    distances = tree.query(points - np.r_[fit.x, z])[0]
    rms = float(np.sqrt(np.mean(distances**2)))
    inliers = float(np.mean(distances < 0.006))
    if rms > 0.006 or inliers < 0.85:
        raise ValueError("Visible surface does not fit the known object")
    return dict(
        position_map=[*fit.x, z],
        quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        surface_fit_rmse_m=rms,
        surface_inlier_fraction=inliers,
        cluster_points=len(points),
        support_plane_z_m=float(plane_height),
        position_source="robokudo_rgbd_model_fit",
        orientation_source="known_upright_training_orientation",
    )


def detect(frame_path, bounds, mesh_path, object_name="bowl"):
    # Reuse upstream import compatibility fix and its actual RoboKudo annotators.
    from cram_vrb_lab.perception import pipeline as upstream
    import robokudo.pipeline
    from robokudo.annotators.core import BaseAnnotator
    from robokudo.cas import CASViews
    from robokudo.types.scene import ObjectHypothesis
    from robokudo.types.annotation import Plane
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
    )
    import open3d as o3d
    from py_trees.common import Status
    import rclpy

    o3d.utility.random.seed(7)
    model = o3d.io.read_triangle_mesh(str(mesh_path))
    np.testing.assert_allclose(
        np.stack([model.get_min_bound(), model.get_max_bound()]), bounds, atol=1e-6
    )
    model_points = np.asarray(model.sample_points_uniformly(40000).points)
    frame = np.load(frame_path)
    transform = frame["map_T_camera"]
    k = frame["intrinsics"]
    rgb = frame["rgb"]

    class FrameReader(BaseAnnotator):
        def update(self):
            cas = self.get_cas()
            cas.set(CASViews.COLOR_IMAGE, rgb[:, :, ::-1].copy())
            cas.set(
                CASViews.DEPTH_IMAGE,
                upstream._depth_metres_to_millimetres(frame["depth_m"]),
            )
            cas.set(CASViews.COLOR2DEPTH_RATIO, (1.0, 1.0))
            cas.set(
                CASViews.CAMERA_INTRINSIC,
                o3d.camera.PinholeCameraIntrinsic(
                    rgb.shape[1], rgb.shape[0], k[0, 0], k[1, 1], k[0, 2], k[1, 2]
                ),
            )
            cas.camera_to_world_transform = HomogeneousTransformationMatrix(
                data=transform
            )
            return Status.SUCCESS

    class CounterCrop(BaseAnnotator):
        def update(self):
            cloud = self.get_cas().get(CASViews.CLOUD)
            points = np.asarray(cloud.points) @ transform[:3, :3].T + transform[:3, 3]
            # Known counter region, independent of the test bowl's actual pose.
            keep = ((points > [-0.65, 7.12, 0.91]) & (points < [0.85, 7.68, 1.20])).all(
                axis=1
            )
            cloud = cloud.select_by_index(np.where(keep)[0])
            if len(cloud.points) < 100:
                raise ValueError("Counter is outside the RGB-D view")
            self.get_cas().set_ref(CASViews.CLOUD, cloud)
            return Status.SUCCESS

    plane = upstream.PlaneAnnotator.Descriptor()
    plane.parameters.distance_threshold = 0.003
    plane.parameters.num_iterations = 100
    clusters = upstream.PointCloudClusterExtractor.Descriptor()
    clusters.parameters.min_cluster_count = 80
    clusters.parameters.dbscan_min_cluster_count = 8
    clusters.parameters.min_on_plane_point_count = 20
    clusters.parameters.eps = 0.012
    pipeline = robokudo.pipeline.Pipeline("GarmiKnownObjectRGBD")
    pipeline.add_children(
        [
            upstream.ClearAnnotatorOutputs(),
            FrameReader("CapturedRGBD"),
            upstream.ImagePreprocessorAnnotator("RGBDPointCloud"),
            CounterCrop("KnownCounterRegion"),
            upstream.PlaneAnnotator(descriptor=plane),
            upstream.PointCloudClusterExtractor(descriptor=clusters),
            upstream.ClusterPoseBBAnnotator(),
        ]
    )
    rclpy.init()
    node = upstream.make_pipeline_node()
    try:
        status = robokudo.utils.tree_execution.run_tree_once(
            pipeline, node, include_gui=False, max_iterations=1000, tick_rate=100
        )
        if status != Status.SUCCESS:
            raise RuntimeError(f"RoboKudo failed: {status}")
        cas = pipeline.cas
        cloud = cas.get(CASViews.CLOUD)
        plane_annotation = cas.filter_annotations_by_type(Plane)[0]
        support = (
            np.asarray(cloud.select_by_index(plane_annotation.inliers).points)
            @ transform[:3, :3].T
            + transform[:3, 3]
        )
        plane_height = float(np.median(support[:, 2]))
        fits = []
        rejected = []
        for hypothesis in cas.filter_annotations_by_type(ObjectHypothesis):
            points = (
                np.asarray(hypothesis.points.points) @ transform[:3, :3].T
                + transform[:3, 3]
            )
            try:
                np.save(frame_path.parent / f"cluster-{hypothesis.id}.npy", points)
                fit = estimate_bowl(points, model_points, bounds, plane_height)
                fit["cluster_id"] = hypothesis.id
                fits.append(fit)
                np.save(frame_path.parent / f"cluster-{hypothesis.id}.npy", points)
            except ValueError as error:
                rejected.append(dict(cluster_id=hypothesis.id, reason=str(error)))
        if len(fits) != 1:
            raise ValueError(
                f"Expected one unambiguous {object_name}, got {len(fits)}; rejected={rejected}"
            )
        fits[0]["object_name"] = object_name
        fits[0]["rejected_clusters"] = rejected
        fits[0]["pipeline"] = [
            "ImagePreprocessorAnnotator",
            "PlaneAnnotator",
            "PointCloudClusterExtractor",
            "ClusterPoseBBAnnotator",
        ]
        # Export the actual input and annotator outputs for the Lab explorer.
        chosen = np.load(frame_path.parent / f"cluster-{fits[0]['cluster_id']}.npy")
        origin = np.asarray(fits[0]["position_map"])
        camera_points = (chosen - transform[:3, 3]) @ transform[:3, :3]
        uv = camera_points @ k.T
        uv = uv[:, :2] / uv[:, 2:]
        def sampled(values, maximum=7000):
            return values[::max(1, int(np.ceil(len(values) / maximum)))].round(6).tolist()
        visual = dict(
            schema_version=1, source="actual_robokudo_run", object_name=object_name,
            image_size=[rgb.shape[1], rgb.shape[0]],
            image_box=[np.min(uv, axis=0).tolist(), np.max(uv, axis=0).tolist()],
            origin_map=origin.tolist(),
            counter=sampled(np.asarray(cloud.points) @ transform[:3,:3].T + transform[:3,3] - origin),
            plane=sampled(support - origin), cluster=sampled(chosen - origin),
            fitted_model=sampled(model_points),
            model_note="Custom fixed-orientation model fit after RoboKudo segmentation",
        )
        (frame_path.parent / "visualization.json").write_text(json.dumps(visual))
        return fits[0]
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=Path("/workspace/rl/assets/manifest.json"),
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path(
            "/workspace/cram-vrb-lab/cognitive_robot_abstract_machine/coraplex/resources/objects/bowl.stl"
        ),
    )
    parser.add_argument(
        "--model-spec", type=Path, help="Known object mesh, bounds and hash"
    )
    args = parser.parse_args()
    started = time.monotonic()
    if args.model_spec:
        import hashlib

        model = json.loads(args.model_spec.read_text())
        mesh = Path(model["mesh"])
        if hashlib.sha256(mesh.read_bytes()).hexdigest() != model["mesh_sha256"]:
            raise ValueError("Perception model mesh changed")
        result = detect(args.frame, model["bounds"], mesh, model["name"])
    else:
        result = detect(
            args.frame,
            json.loads(args.asset_manifest.read_text())["bowl_bounds"],
            args.mesh,
        )
    result["processing_seconds"] = time.monotonic() - started
    result["capture_file"] = str(args.frame)
    (args.frame.parent / "estimate.json").write_text(json.dumps(result, indent=2))
    print("ROBOKUDO_OBJECT", json.dumps(result), flush=True)
