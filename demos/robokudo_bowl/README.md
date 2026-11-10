# RoboKudo bowl grasp

This experiment connects GARMI's simulated head RGB-D camera to the existing CRAM
station selection and `RLPickUpAction`. The trained policy moves the base and right
arm, then the existing controller closes the fingers and lifts the bowl.

The experiment lives on `codex/robokudo-bowl-20261110`. The previous Lab stays at
`codex/lab-release-preview-20261106` (`24f9b5b` in the RL repository), also marked by
`local/pre-robokudo-20261110`. The existing Lab entry points are unchanged. This standalone test records video
and action logs; it does not start NeemHub.

## Data flow

1. Capture aligned RGB and optical-axis depth at the robot's head. The lens position
   follows the upstream GARMI mounting offset. A fixed 30-degree downward optical
   tilt accommodates the training asset's frozen head joints. Camera-to-map
   calibration comes from robot kinematics, not object state.
2. Run the installed RoboKudo `ImagePreprocessorAnnotator`, `PlaneAnnotator`,
   `PointCloudClusterExtractor`, and `ClusterPoseBBAnnotator`. A known counter region
   excludes the rest of the apartment. The worker receives only RGB-D, camera
   calibration and the known bowl mesh; it has no simulator object handle.
3. Fit the known bowl surface to its observed cluster. The support plane and mesh
   bottom determine height; surface alignment determines horizontal translation.
   This asset is a rounded square bowl. Orientation is the known upright training
   orientation, not a circular-rim estimate or general six-degree pose estimator.
4. Publish that estimated pose into the normal CRAM/RL bridge. CRAM uses it to
   select a station and an allowed grasp reference. The same estimate provides
   object-relative features and reference targets for the trained policy and the
   fixed finishing controller.
5. Evaluate the physical lift with the existing contact, height, slip and collision
   checks. Score localization against simulator truth in a separate file.

The initial observation is held fixed while the robot approaches a stationary
bowl. During lift, the estimate follows the measured gripper transform. This first
experiment uses one known upright bowl and one attempt; an unobserved displacement
or an uncertain/missing detection requires another observation rather than a retry
with simulator coordinates. Physical contact/safety checks and success scoring
still use simulation measurements. The trained checkpoint and training files are
unchanged.

## Run in the native environment

Use a separate container of the existing validated Isaac Sim 6.1 deployment image,
with the runtime files in `/workspace`. The experiment uses ROS domain 59. It needs
the same packaged assets and `policy-7500` bundle as the previous demo; RoboKudo and
Open3D are already installed in that image's CRAM Python environment.

```bash
/opt/conda/bin/python /workspace/demos/robokudo_bowl/run.py \
  --out /perception-results/my-run \
  --bowl-map 0.08 7.28 \
  --base-map -0.1 5.9 1.5707963267948966 \
  --record
```

Use a new output directory per run. `--record` saves a third-person video. The
supervisor starts the simulator, waits for a valid detection, starts Giskard, runs
the CRAM action and closes only its own process groups. No grasp model is trained. The standalone task uses `GraspDemo` directly; this
also avoids an observed lock inversion between NeemHub snapshot publication and
CRAM world-state updates during symbolic attachment. The original live Lab is
left unchanged.

For camera-only checks, run `sim.py --capture-only --headless --out ...` through
`scripts/local_python.sh`, then run `detect.py <out>/perception/rgbd.npz` with the
CRAM Python environment and ROS setup sourced.

## Evidence

- `sim/perception/rgb.png`, `rgbd.npz`: the camera input.
- `sim/perception/estimate.json`: fitted pose, cluster and fitting diagnostics.
- `sim/localization-score.json`: separately recorded physical pose and error.
- `sim/input-isolation.json`: observation computation still works when the
  simulator object handle is replaced by an unusable sentinel; moving only the
  perceived pose changes the observation. The fixed finish uses the same view.
- `sim/policy-inputs.jsonl`: actual object/grasp inputs for each policy tick.
- `task/rl-grasp-report.json`, `task/result.json`: CRAM station, reference and grasp outcome.
- `sim/replay.mp4`: optional recording; `run.json`: supervisor outcome.

Run the geometry tests with `python -m pytest demos/robokudo_bowl/tests -q` in an
environment containing NumPy, SciPy and pytest. Runtime input-isolation checks run
before every grasp.
