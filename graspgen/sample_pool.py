"""Broaden the two YCB grasp pools using independent seeds and input yaw rotations."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

from generate import ROOT, OBJECTS, sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/graspgen-ycb-40refs')
    parser.add_argument('--objects', nargs='+', choices=OBJECTS, default=list(OBJECTS))
    parser.add_argument('--mesh', type=Path, help='Custom mesh for a single selected object.')
    parser.add_argument('--fresh', action='store_true', help='Generate all candidates anew; do not reuse the original pool.')
    parser.add_argument('--batches', type=int, default=16)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--orientation-augmentation',choices=['yaw','cube'],default='yaw',
        help='cube uses the 24 proper cube rotations to expose orientation modes missed by yaw-only inference.')
    args = parser.parse_args()
    if args.mesh and len(args.objects)!=1:parser.error('--mesh requires exactly one --objects entry')
    if args.mesh and not args.fresh:parser.error('--mesh requires --fresh to avoid mixing candidates from different geometry')
    if args.batches<1:parser.error('--batches must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_grasp_cfg(str(ROOT/'.runtime/graspgen-models/checkpoints/graspgen_franka_panda.yml'))
    sampler = GraspGenSampler(cfg)
    provenance = json.loads((ROOT/'runs/graspgen-ycb/summary.json').read_text())
    info = dict(source_run='runs/graspgen-ycb', model_revision=provenance['model_revision'],
        graspgen_commit=provenance['graspgen_commit'], checkpoints=provenance['checkpoints'],
        description=f'{args.batches} batches of 400; original candidates reused={not args.fresh}; poses mapped back to input mesh frame',
        orientation_augmentation=args.orientation_augmentation,objects={})
    for name in args.objects:
        out = args.out/name
        out.mkdir(exist_ok=True)
        target = out/'candidate_pool.npz'
        if target.exists():
            raise FileExistsError(target)
        poses=[];scores=[];batches=[];original_count=0;input_clouds=[]
        if not args.fresh:
            original = np.load(ROOT/'runs/graspgen-ycb'/name/'all_candidates.npz')
            poses = [original['T_object_graspgen']]; scores = [original['scores']]
            original_count=len(poses[0]);batches = [np.full(original_count, -1, dtype=int)]
        mesh_path=args.mesh.resolve() if args.mesh else ROOT/'assets/ycb'/name/'google_16k/textured.obj'
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        records = []
        for batch in range(args.batches):
            seed = args.seed_start + batch; yaw = (batch % 4) * 90
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            points, _ = trimesh.sample.sample_surface(mesh, 2048, seed=seed)
            aug = np.eye(4)
            aug[:3,:3] = (Rotation.from_euler('z', yaw, degrees=True).as_matrix()
                if args.orientation_augmentation=='yaw' else Rotation.create_group('O')[batch%24].as_matrix())
            aug[:3,3] = mesh.bounds.mean(0) - aug[:3,:3] @ mesh.bounds.mean(0)
            points = np.asarray(trimesh.transform_points(points, aug), dtype=np.float32)
            input_clouds.append(points)
            g, s = GraspGenSampler.run_inference(points, sampler, grasp_threshold=-1,
                num_grasps=400, topk_num_grasps=400, min_grasps=1, max_tries=1, remove_outliers=False)
            if not len(g):
                raise RuntimeError(f'Empty batch {name} {batch}')
            g = np.linalg.inv(aug)[None] @ g.cpu().numpy()
            poses.append(g); scores.append(s.cpu().numpy()); batches.append(np.full(len(g), batch, dtype=int))
            records.append(dict(batch=batch, seed=seed,
                input_yaw_degrees=yaw if args.orientation_augmentation=='yaw' else None,
                input_rotation_matrix=aug[:3,:3].tolist(),count=len(g)))
            print(name, records[-1], flush=True)
        np.savez_compressed(target, T_object_graspgen=np.concatenate(poses),
            scores=np.concatenate(scores), batch_id=np.concatenate(batches))
        np.savez_compressed(out/'input-pointclouds.npz',points=np.asarray(input_clouds))
        info['objects'][name] = dict(batches=records, original_count=original_count,mesh=str(mesh_path),mesh_sha256=sha(mesh_path),
            total=sum(map(len, poses)), pool_sha256=sha(target), original_pool_sha256=None if args.fresh else sha(ROOT/'runs/graspgen-ycb'/name/'all_candidates.npz'))
        (args.out/'pool-provenance.json').write_text(json.dumps(info, indent=2)+'\n')


if __name__ == '__main__':
    main()
