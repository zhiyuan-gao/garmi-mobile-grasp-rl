# Validation and provenance

## Original continuous demonstration — 2026-10-22

The recorder in this directory is byte-identical to the source used for the
completed full-apartment video. The task used `position-1-body-16`, the published
mug PPO update-6500 checkpoint and the two camera settings in the README.
The original run did not use `--debug-resume` or restore a saved simulation state.

| Measurement | Result |
| --- | --- |
| Simulated duration | 55.7 seconds |
| Final mug bounds | Inside the drawer |
| Final mug tilt | 0.404 degrees |
| Final drawer opening | 0.000000323 m |
| Final mug speed | 0.0000498 m/s |
| Export | 1600 × 900, 30 fps, 4×, 13.933333 seconds |

The original MP4 passed full FFmpeg decoding and Chromium playback. Its hash,
checkpoint hash, recorder hash and final measurements are retained in
[`configs/recorded-validation.json`](configs/recorded-validation.json). The video,
models, assets and raw traces are external artifacts and are not included in Git.
This single configured demonstration does not establish a general success rate.

## Standalone integration checks — 2026-10-23

- Python syntax compilation and shell syntax checks.
- The extracted GARMI kinematics match the original implementation exactly for
  forward kinematics at 100 joint configurations and an inverse-kinematics solve.
- The exporter rejects debug-resumed runs, incomplete runs and mug-outside runs
  before creating an output directory.
- Both all-40 reference PNGs were regenerated through this directory's renderer.
  Mesh hashes, reference bank hashes, all 40 IDs and depth-audit counts match the
  original corrected images. Repeated EGL rendering differs by at most one
  intensity level in a small number of pixels.
- A fresh asset build exposed mesh basenames containing hyphens, which the URDF
  importer incorrectly used as USD prim identifiers. The builder now creates
  stable aliases with valid identifiers under the generated `mesh_aliases/`
  directory. The meshes themselves are unchanged. The rebuilt passive drawer
  contains the original 11 collision pieces.

- A new complete recording from freshly built assets passed in the standalone
  directory: RL grasp, contact-driven opening, placement and closure. The run
  finished at 55.7 simulated seconds; the mug was inside and stable, with
  drawer opening below 1 mm. No debug resume was used. Final measurements are
  included under `standalone_integration_check` in the validation JSON.

Generated verification artifacts remain in ignored `assets/`, `runs/` and
`exports/` directories. The launcher reuses the frozen release and existing
Apptainer runtime; it does not depend on the original VLA checkout.
