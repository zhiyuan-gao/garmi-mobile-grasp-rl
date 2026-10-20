"""Offline GraspGen candidates in original YCB mesh coordinates; no RL training."""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.dataset.eval_utils import save_to_isaac_grasp_format

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = ("019_pitcher_base", "025_mug")


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(path.read_bytes()).hexdigest()


def preview(mesh, hand, poses, scores, clearances, out, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import plotly.graph_objects as go

    color = "#426aab" if "pitcher" in name else "#ab4840"
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    for j, pose in enumerate(poses[:6]):
        ax = fig.add_subplot(2, 3, j + 1, projection="3d")
        hv = trimesh.transform_points(hand.vertices, pose)
        ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=color, linewidths=0, alpha=.65))
        ax.add_collection3d(Poly3DCollection(hv[hand.faces], facecolors="#e9b139", linewidths=0, alpha=.95))
        verts = np.vstack([mesh.vertices, hv])
        center = (verts.min(0) + verts.max(0)) / 2
        radius = np.ptp(verts, axis=0).max() * .55
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=-60)
        ax.set_axis_off()
        ax.set_title(f"Rank {j+1} | score {scores[j]:.3f}\nplane clearance {clearances[j]*1000:.1f} mm", fontsize=10)
    fig.suptitle(f"{name} | GraspGen / Franka Panda | original scale", fontsize=15)
    fig.text(.5, .015, "Yellow: open gripper. Candidates only; no GARMI reachability or physical grasp validation.", ha="center", fontsize=10)
    fig.subplots_adjust(left=0, right=1, bottom=.05, top=.91, wspace=0, hspace=.13)
    fig.savefig(out / "top6.png", dpi=135)
    plt.close(fig)

    # Self-contained HTML: offline rotation/zoom and a candidate selector.
    def trace(vertices, faces, label, col, visible=True):
        return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=label,
            color=col, opacity=1, visible=visible, flatshading=False)
    traces = [trace(mesh.vertices, mesh.faces, name, color)]
    for j, pose in enumerate(poses[:20]):
        traces.append(trace(trimesh.transform_points(hand.vertices, pose), hand.faces,
            f"Rank {j+1}, score {scores[j]:.3f}", "#e9b139", j == 0))
    buttons = []
    for j in range(len(traces)-1):
        visible = [True] + [i == j for i in range(len(traces)-1)]
        buttons.append(dict(label=f"#{j+1}: score {scores[j]:.3f}", method="update", args=[{"visible": visible}]))
    html = go.Figure(traces)
    html.update_layout(title=f"{name}: GraspGen candidates (not physically validated)",
        scene=dict(aspectmode="data", xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
        updatemenus=[dict(buttons=buttons, x=0, y=1.12)], margin=dict(l=0, r=0, b=0, t=110))
    html.write_html(out / "viewer.html", include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
    parser.add_argument("--num-grasps", type=int, default=400)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/graspgen-ycb")
    parser.add_argument("--config", type=Path, default=ROOT / ".runtime/graspgen-models/checkpoints/graspgen_franka_panda.yml")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        raise FileExistsError("Choose a new --out directory to preserve the existing run")
    cfg = load_grasp_cfg(str(args.config))
    sampler = GraspGenSampler(cfg)
    hand = get_gripper_info("franka_panda").collision_mesh
    metadata = dict(graspgen_commit=subprocess.check_output(["git", "-C", str(ROOT / "third_party/GraspGen"), "rev-parse", "HEAD"], text=True).strip(),
        model_revision=(ROOT / ".runtime/graspgen-models/revision.txt").read_text().strip(),
        checkpoints={Path(p).name: sha(Path(p)) for p in [cfg.eval.checkpoint, cfg.discriminator.checkpoint]},
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), planner="diffusion", gripper="franka_panda",
        frame="T_object_graspgen: maps gripper-local points to original OBJ coordinates; meters; +Z approach, X closing axis",
        validation="Candidate generation only. No GARMI IK, object collision, approach-path or physical lift test. Plane clearance is a diagnostic only.",
        objects={})
    for name in args.objects:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        mesh_path = ROOT / "assets/ycb" / name / "google_16k/textured.obj"
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        points, _ = trimesh.sample.sample_surface(mesh, 2048, seed=args.seed)
        torch.cuda.synchronize(); start = time.perf_counter()
        poses_t, scores_t = GraspGenSampler.run_inference(np.asarray(points, dtype=np.float32), sampler,
            grasp_threshold=-1, num_grasps=args.num_grasps, topk_num_grasps=args.num_grasps,
            min_grasps=1, max_tries=1, remove_outliers=False)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        if len(poses_t) == 0:
            raise RuntimeError(f"No candidates for {name}")
        poses = poses_t.cpu().numpy(); scores = scores_t.cpu().numpy()
        order = np.argsort(-scores); poses = poses[order]; scores = scores[order]
        assert np.isfinite(poses).all() and np.isfinite(scores).all()
        assert np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5)
        rotations = poses[:, :3, :3]
        assert np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
        assert np.allclose(np.linalg.det(rotations), 1, atol=1e-4)
        clearance = np.array([trimesh.transform_points(hand.vertices, p)[:, 2].min()-mesh.bounds[0, 2] for p in poses])
        out = args.out / name; out.mkdir(exist_ok=True)
        np.savez_compressed(out / "all_candidates.npz", T_object_graspgen=poses, scores=scores, point_cloud=points, plane_clearance_m=clearance)
        k = min(args.topk, len(poses))
        quat = Rotation.from_matrix(rotations[:k]).as_quat()[:, [3, 0, 1, 2]]
        records = [dict(rank=j+1, score=float(scores[j]), position_m=poses[j, :3, 3].tolist(),
            quaternion_wxyz=quat[j].tolist(), T_object_graspgen=poses[j].tolist(),
            plane_clearance_m=float(clearance[j])) for j in range(k)]
        (out / "top_grasps.json").write_text(json.dumps(dict(frame=metadata["frame"], grasps=records), indent=2)+"\n")
        save_to_isaac_grasp_format(poses[:k], scores[:k], str(out / "top_grasps.yaml"))
        metadata["objects"][name] = dict(mesh=str(mesh_path.relative_to(ROOT)), mesh_sha256=sha(mesh_path),
            dimensions_m=mesh.extents.tolist(), scale=1.0, seed=args.seed, sampled_points=len(points),
            candidates=len(poses), exported_topk=k, inference_seconds=elapsed,
            score_min=float(scores.min()), score_max=float(scores.max()), score_median=float(np.median(scores)),
            scores_ge_05=int((scores >= .5).sum()), topk_clear_of_plane=int((clearance[:k] >= 0).sum()))
        # Save numeric results before rendering, so visualisation cannot lose an inference run.
        (args.out / "summary.json").write_text(json.dumps(metadata, indent=2)+"\n")
        print(name, metadata["objects"][name], flush=True)
        preview(mesh, hand, poses[:k], scores[:k], clearance[:k], out, name)


if __name__ == "__main__":
    main()
