# RoboKudo mug validation — 2026-11-10

The isolated `codex/robokudo-mug-20261110` branch extends the saved bowl experiment
with a mug-specific CRAM body, policy binding and verified inference bundle.
See [run instructions](../demos/robokudo_mug/README.md).

## Final configuration

- Existing mug update-6500 checkpoint, SHA-256
  `b6ae2305fcbb353d88801ca3f9817b74665f34f9d8238b6fcd5a96d41e082ce2`.
- Existing `household_rl` implementation, normalizer and controller settings.
- Reference `mug-v3-ref025`, known upright orientation, original placement mask.
- Head RGB-D camera at a fixed 30-degree downward tilt and 70-degree horizontal
  field of view. Calibration comes from robot kinematics and camera intrinsics.
- RoboKudo plane segmentation and clustering, followed by known-model surface
  alignment. The estimated translation drives CRAM station selection and RL
  object/reference observations. This is not arbitrary handle-pose estimation.
- Simulation contact/safety measurements and physical scoring remain separate
  from the perception-fed policy geometry. No retraining or physical attachment.

## Completed tests

Coordinates are map metres; initial base heading is pi/2 radians.

| Run | Mug XY | Base initial XY | 3D localization error | Lift | Hold | RL base path |
| --- | --- | --- | --- | --- | --- | --- |
| grasp-05 | 0.162, 7.262 | 0.08, 5.90 | 2.52 mm | 11.79 cm | 3.10 s | 36.54 cm |
| grasp-06 | 0.160, 7.260 | 0.00, 5.80 | 3.04 mm | 11.76 cm | 3.05 s | 36.74 cm |

Both completed CRAM preparation/navigation, learned approach, fixed close/lift,
physical hold verification and CRAM handback. Final contact was present, counter
support absent and collision flags zero. Base path is accumulated from recorded
states while the controller owner is RL; it excludes CRAM navigation.

The two object positions differ by only 2 mm per axis inside the original 1 cm
placement cell. Robot starts differ by about 13 cm; CRAM navigates to a station
before handing control to RL. These are integration checks, not broad placement
or grasp-success benchmarks.

## Development outcomes retained

- grasp-01 stopped before scene execution because the launcher selected the
  older bowl controller module. Selecting the existing household implementation
  restored the mug controller settings without altering the policy.
- grasp-02 localized the mug but stopped during CRAM registration: an inherited
  freshness callback recognized only the bowl message name. The mug task now
  recognizes its own object name.
- grasp-03 completed successfully with the original narrower camera view:
  localization error 1.32 mm, lift 11.80 cm and stable hold 3.10 seconds.
- grasp-04 stopped at perception because the closer initial viewpoint cropped
  the mug. The detector rejected its incomplete extent. Fixed wider camera
  calibration resolved this in grasp-05; no simulator-coordinate fallback.

All trials and failures are retained under
`/home/zgao/FAME_RL/deployment/robokudo-mug-20261110/results/`.
`index.html` shows both final recordings, trimmed of initial waiting and sped
up to about nine seconds, plus links to full videos, images and diagnostics.

## Verification and preservation

- 33 tests passed, including accepting a mug skill for the mug task and rejecting
  a bowl skill before motion.
- Every successful physical run passed policy-input isolation checks.
- Using the bowl mesh on a mug RGB-D capture is rejected without an estimate.
- 228 mapped runtime source files pass package verification.
- The previous bowl worktree remains clean at `538ce33`; the older Lab worktrees
  remain clean at RL `24f9b5b` and deployment `3046605`.

The initial observation is frozen during approach. Unexpected object movement,
arbitrary handle rotation and continuous visual correction still require more
work. This standalone experiment stops after grasping and uses no NeemHub live
UI. It does not change the long household task, the VRB page or remote branches.
