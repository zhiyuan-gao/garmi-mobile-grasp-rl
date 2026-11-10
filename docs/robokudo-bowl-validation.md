# RoboKudo bowl validation — 2026-11-10

The isolated `codex/robokudo-bowl-20261110` experiment uses head-camera RGB-D
localization for CRAM station selection, reference targets and learned grasp
observations. The policy checkpoint is unchanged. Implementation: `4ec6da1`.
See [the experiment README](../demos/robokudo_bowl/README.md) for execution and
the exact perception assumptions.

## Completed physical tests

Coordinates are in the existing apartment map, in metres. Robot heading starts
at pi/2 radians in both configurations.

| Run | Bowl XY | Robot initial XY | Localization error | Lift | Stable hold |
| --- | --- | --- | --- | --- | --- |
| grasp-01 | 0.20, 7.40 | 0.00, 5.80 | 0.75 mm | 11.87 cm | 3.35 s |
| grasp-02 | 0.08, 7.28 | -0.10, 5.90 | 1.88 mm | 12.04 cm | 3.20 s |
| grasp-04 | 0.08, 7.28 | -0.10, 5.90 | 4.38 mm | 11.76 cm | 3.20 s |

These are three completed trials at two unique initial configurations, not a
statistical success-rate benchmark. All three report physical contact, no
support from the counter, and no collision flag at completion. Localization
errors compare the initial estimate against simulator truth logged separately.

Run grasp-03 also physically grasped and held the bowl, but the task stopped
after a NeemHub/world-state lock inversion during post-grasp synchronization.
A stack dump was saved. The final standalone runner uses `GraspDemo` directly,
without NeemHub, and completed grasp-04. This does not fix or modify the older
interactive Lab; adding perception to that live UI still needs validation.

## Checks

- 31 geometry, action, adaptive selection and planning-probe tests passed in the
  native CRAM environment.
- Runtime policy observations and fixed-finish geometry work with the simulator
  object handle replaced by an unusable sentinel. Perturbing only the perceived
  pose changes policy inputs.
- Empty depth is rejected by the actual perception worker, with no estimate or
  fallback to simulator object coordinates.
- All 217 mapped runtime source files passed package verification.

This is an initial observation of a known upright bowl at its training
orientation. Position is measured; arbitrary orientation is not inferred.
During approach the estimate is held fixed, and during lift it follows gripper
kinematics. Safety/contact checks and physical success scoring still use the
simulation. Unexpected displacement requires a new observation.

## Local evidence and preservation

Evidence root on the development machine:
`/home/zgao/FAME_RL/deployment/robokudo-bowl-20261110/results/`.
Open `index.html` for the recording, camera input and linked diagnostics.
The full recording is `grasp-04/sim/replay.mp4`; the preview is nine seconds.

The prior RL worktree remains at `24f9b5b`, with backup tag
`local/pre-robokudo-20261110`. The prior deployment worktree remains at
`3046605`. Both were clean after the experiment. No upstream push or VRB
website update was performed.
