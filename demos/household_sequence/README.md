# Continuous GARMI household demonstration

One invocation runs a single task in the full GARMI apartment:

1. Pick up the bowl from the left kitchen counter and place it on the dining table.
2. Return to the kitchen counter and pick up the mug that has remained there since the start.
3. Open the upper kitchen drawer with the left hand, place the mug inside with the right hand, and close the drawer with the left hand.

The bowl and mug both start on the left kitchen counter, spaced apart. Both objects and the passive drawer are present throughout. The simulator is
initialized once and keeps running between stages. The bowl stays on the table
while the robot handles the mug. The output is one continuous recording.

## Controllers

The bowl segment reuses the existing CRAM/Giskard counter-to-table routine and
bowl update-7500 policy. `bowl_stage.py` adds the counter mug to the planning world.
The mug segment uses the published mug update-6500 policy with reference
`mug-v3-ref025`. Its original fixed closing/lifting controller is retained.
The mug retains its original counter height, orientation and policy movement bounds; its starting position and approach base pose are translated sideways together to separate it from the bowl.

The new connecting base motions use a slow waypoint velocity controller with
the existing acceleration limits. Drawer manipulation reuses the standalone
mug demo's Cartesian IK sequence and passive drawer: the left fingers pull and
push it through contact. `backend.py` owns these connecting motions.

## Run locally

Use the existing Isaac Sim 5.1 / Isaac Lab 2.3.2 Docker runtime and the validated
bowl and mug workspaces. Paths are supplied explicitly:

```bash
export GARMI_BOWL_WORKSPACE=/path/to/FAME_RL
export GARMI_MUG_WORKSPACE=/path/to/mug-ppo6500-workspace
export GARMI_ISAACLAB=/path/to/IsaacLab
export GARMI_DEMO_IMAGE=garmi-rl-fork:validation

bash demos/household_sequence/docker.sh prepare
bash demos/household_sequence/docker.sh preview
bash demos/household_sequence/docker.sh run
```

`prepare` calls the existing mug drawer asset builders and writes their outputs
under this demo's `assets/`. `preview` loads the initial scene for inspection.
`run` starts one simulation, one planning server and the bowl task, then continues
with the mug in that same simulation. Each invocation owns its Docker container
and child processes. Run outputs are timestamped beneath this demo's `runs/`.

## Outputs and checks

- `report.json`: overall completion or failure.
- `bowl/rl-transport-report.json`: CRAM motions and physical bowl placement checks.
- `sim/sequence.json`: stage transitions, mug grasp result and final verification.
- `sim/states.jsonl`: measured robot and object state over time.
- `sim/replay.mp4`: the continuous recording.
- `sim/initial.png`, `sim/final.png` or `sim/failed.png`: inspection images.

The second stage starts only after the bowl routine confirms physical release,
support, stability and completion. The final check requires the mug inside the
closed drawer, stationary, and the bowl still near its verified table position.
A failed grasp or motion stops the task and retains its logs and recording.

The existing standalone bowl and mug demo files are reused through imports;
this directory contains the new task entry and its scene-specific adaptations.

After a successful run, export a 4× video and local review page:

```bash
python3 demos/household_sequence/export_review.py \
  demos/household_sequence/runs/<run-directory>
```

The exporter checks completion and final verification before producing
`review/index.html` and `review/household-4x.mp4`.
