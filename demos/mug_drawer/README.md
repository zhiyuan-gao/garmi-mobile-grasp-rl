# Mug body grasp and dual-arm drawer manipulation

The right hand grasps a mug with the published PPO update-6500 policy. Scripted
Cartesian IK then moves the arms: hold the mug, open the upper drawer with the
left hand, place and release the mug with the right hand, clear the drawer, and
close it with the left hand. The drawer has a passive prismatic joint. Object
support and drawer motion use physical contact, without attachments or a drawer
motor. This is one configured demonstration, not an evaluation of a general
drawer policy. The post-grasp actions do not use CRAM or Giskard.

The full-scene recording uses the original GARMI apartment. A counter-corner
camera shows the grasp; an elevated side camera shows the drawer sequence. The
camera changes during the same continuous simulation, without resetting state.

## Existing resources

1. Prepare the frozen mug release and import its assets using the
   [mug checkpoint guide](../../docs/mug-ppo-6500.md). The default location is
   `.runtime/mug-ppo6500` at the repository root. The recorder checks the published
   checkpoint SHA256 before inference and reads its original PPO configuration.
   Expert training data are not needed for this inference-only demonstration.
2. Prepare the existing Apptainer runtime with
   [the repository setup instructions](../../docs/usage.md). The launcher expects
   `.runtime/container`, `third_party`, and the original GARMI description and
   apartment inside the runtime at `/home/jovyan/cram-vrb-lab`. It uses the existing
   Isaac Sim 5.1.0 / Isaac Lab 2.3.2 environment. It does not install another copy.
3. Export videos on the host with Python and FFmpeg (`ffmpeg` on PATH, or pass
   `--ffmpeg /path/to/ffmpeg`). Asset builders and the recorder run in the simulator.

To reuse resources in another checkout, set absolute paths before running:

```bash
export GARMI_DRAWER_WORKSPACE=/path/to/unpacked/mug-ppo6500
export GARMI_DRAWER_RUNTIME=/path/to/existing/.runtime/container
export GARMI_DRAWER_THIRD_PARTY=/path/to/existing/third_party
export GARMI_DRAWER_GPU=0
```

`scripts/sim.sh` supports Apptainer. It mounts the release at `/workspace` and
this demo directory at `/demo`, so paths inside the simulation are independent
of the host checkout location. All commands below start at the repository root.

## Build, record, and export

```bash
bash demos/mug_drawer/scripts/sim.sh scripts/prepare_mug_drawer.py
bash demos/mug_drawer/scripts/sim.sh scripts/prepare_mug_drawer_apartment.py
bash demos/mug_drawer/scripts/sim.sh scripts/record_mug_drawer.py \
  --headless --full-scene --case position-1-body-16 \
  --camera-eye -0.65 2.0 2.4 --camera-target 0.5 0.2 0.95 \
  --grasp-camera-eye 1.3 1.6 2.4 --grasp-camera-target 0.55 0.2 1.0 \
  --out runs/mug-drawer-apartment
python3 demos/mug_drawer/scripts/export_mug_drawer.py \
  demos/mug_drawer/runs/mug-drawer-apartment --mp4-only \
  --out demos/mug_drawer/exports/mug-drawer-apartment
```

The export contains only `mug-to-drawer-full-scene-4x.mp4`. Recording and export
directories must not already exist. Generated assets, recordings and exports
are ignored by Git. The original normal-speed `raw.mp4` remains in the run
directory until you remove it after checking the export.

The recorder saves its executed source, selected case, contact trace, event
snapshots, checkpoint hash and `report.json`. Export requires a successful RL
grasp, completed continuous run, drawer opening, mug placement and drawer closure.
Debug-resumed runs are rejected. `--preview-only` supports camera inspection;
`--preview-state` and `--debug-resume` are diagnostic options, not continuous demos.

## All-40 reference images

The optional renderer puts all 40 reference targets for each object into a single
PNG without individual numbers. It needs NumPy, SciPy, trimesh, pyrender, Pillow,
PyYAML and an EGL OpenGL context; it does not start Isaac Sim.

```bash
python3 demos/mug_drawer/scripts/render_reference_galleries.py \
  --mug-workspace /path/to/unpacked/mug-ppo6500 \
  --bowl-workspace /path/to/bowl-policy-release \
  --out demos/mug_drawer/exports/reference-grasps-all40
```

The bowl source must contain `results/cram-rl-integration/policy-7500/reference-bank.json`
and its referenced assets, as in the existing bowl deployment release. The mug
source is the imported update-6500 runtime. Both meshes and their gripper outlines
share a depth buffer. These show open-gripper target poses, not measured closed
contact states. Source hashes and all reference IDs are saved alongside the PNGs.

See [validation and provenance](VALIDATION.md) for the recorded demonstration.
