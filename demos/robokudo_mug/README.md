# RoboKudo mug grasp

GARMI observes the known YCB mug with a simulated head RGB-D camera. RoboKudo
segments the counter and object, and the known mug mesh is aligned to the visible
surface. CRAM selects a station using that position and invokes `RLPickUpAction`.
The frozen mug update-6500 policy controls the base and right arm; its existing
controller closes the fingers and lifts the mug.

This experiment is on `codex/robokudo-mug-20261110`, based on the separately saved
bowl perception branch. The original Lab and bowl worktrees are preserved.

## Initial scope

Use the training mug with its known upright, map-yaw-zero orientation. Only
translation is estimated; handle orientation is fixed for these tests. A fixed
70-degree horizontal camera field of view keeps the complete mug visible from
the tested starting positions; its measured intrinsics are passed to RoboKudo. The
initial RGB-D estimate drives station selection, object-relative observations,
reference transformation and fixed-finish geometry. It is held fixed until lift,
then follows gripper kinematics. Physical safety and success checks still use
simulation measurements. Missing or ambiguous perception stops execution.

The first grasp reference is `mug-v3-ref025`, the body grasp used in the household
demo. The original spatial mask permits it in the small `local-00` placement
cell, X `[0.155, 0.165]`, Y `[7.255, 7.265]` metres. Tests stay within this cell;
changing robot starts does not imply large-area object-placement generalization.
Other counter locations require their own allowed references and validation.

## Run

Use an isolated copy of the validated native Isaac Sim 6.1 runtime with the
household mug assets, checkpoint and the runtime source map installed. The mug
launcher selects the existing `household_rl` implementation used by the long demo,
including its wrist-contact policy and fixed-finish configuration. No
training is performed. The source map also includes the original mug mask and
its small provenance files. The runner uses ROS domain 60 and does not start
NeemHub or change the Lab web page.

```bash
/opt/conda/bin/python /workspace/demos/robokudo_mug/run.py \
  --out /perception-results/mug-test \
  --mug-map 0.16 7.26 \
  --base-map 0.0 5.8 1.5707963267948966 \
  --record
```

Each run needs a fresh output directory. `prepare.py` verifies the published
checkpoint, assets and original mask provenance, relocates resource paths, and
exports the deterministic actor with its frozen normalizer. Mask cells, policy
weights and control settings remain unchanged. Prepared artifacts are recorded
under `sim/bundle`.

Outputs follow the [bowl experiment](../robokudo_bowl/README.md): camera input,
pose estimate, separate localization scoring, policy-input isolation checks,
CRAM decisions, physical result and optional video. `run.json` reports whether
the entire task completed, independently of a physical grasp success event.
