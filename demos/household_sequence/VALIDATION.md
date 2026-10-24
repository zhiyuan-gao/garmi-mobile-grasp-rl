# Continuous task validation

Validated locally on 2026-10-23 with Isaac Sim 5.1 and the full GARMI apartment.
Run: `runs/garmi-household-run-20261023-023241`.

Both objects start on the left kitchen counter. One uninterrupted task delivered
the bowl to the dining table, returned for the mug, opened the upper drawer with
the left hand, placed the mug inside with the right hand, and closed the drawer.

Measured results:

- Overall completion and final physical verification: passed.
- Task executions: 1; scene initializations: 1.
- Simulation duration: 165.95 seconds.
- Mug grasp: lifted 12.4 cm and held for 2.05 seconds; grasp collision metric zero.
- Drawer opening before placement: 37.1 cm.
- Final mug mesh bounds lie inside the closed drawer; speed 0.00031 m/s.
- Final drawer opening: below 0.001 mm.
- Bowl movement after verified table delivery: below 0.001 mm.

The connecting motion preserves the base heading reached by the mug policy.
Placement includes a separate 6 cm motion toward the drawer front before
descending, giving the right wrist clearance from the counter edge. Earlier
diagnostic runs identified these two integration issues; the run above executes
the corrected sequence continuously from its initial scene.

The recording and measurements are in the run directory:
`review/index.html`, `sim/replay.mp4`, `sim/sequence.json`,
`bowl/rl-transport-report.json`, and `report.json`.
The task handoff checks also pass their two unit tests, and Python compilation
and shell syntax checks pass. Existing tracked task files are unchanged.

This records one successful local demonstration. The VRB deployment and website
have not been updated by this change.

## Native Lab integration (2026-10-24)

Source commit: `30c1cb568da3b6f836c0f9c10a4e7fe131e0fbcd`.
Deployment candidate: `zhiyuan-gao/cram-vrb-lab@75ee1c5e3b4a598c80ccca8bc840f19ff29e0673`.
Local run using the deployment's pinned Isaac Sim 6.1 base image:
`results/cram-rl-web/20261024-140516-039b`.

- Complete physical task: passed, one scene initialization, 159.1 simulation seconds.
- Bowl released onto the dining table, supported and stationary; target error 2.25 mm.
- Mug grasp succeeded: lift 12.27 cm, hold 2.05 seconds, no forbidden contact.
- Drawer opened approximately 37 cm and closed to within 0.001 mm.
- Final mug mesh bounds inside the drawer, stability and bowl-retention checks passed.
- NVIDIA desktop viewer connected automatically and displayed rendered simulation frames.
- Workspace displayed all 13 task stages, ending at Completed while keeping both panes open.
- Refreshing the Workspace reattached to the existing run without a scene reset.
- 299 measured synchronization records reached the final 0.45 seconds of the task.
- Direct `/state` inspection of NeemHub confirmed the measured mug pose and drawer
  joint opening were published to the viewer, alongside the live robot state.

Evidence remains in the local run directory: `sim/sequence.json`,
`task/rl-transport-report.json`, `task/live-sync.jsonl`, `live-viewer-check.json`,
`workspace-completed.png`, and `viewer-connect.log`. Public Binder validation is
tracked separately from this local deployment-image test.
