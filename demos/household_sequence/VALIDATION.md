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
