# CRAM mobile pick-up action

`RLPickUpAction` makes the trained GARMI mobile grasp skill an action in a CRAM
plan. It runs RL approach, fixed closing and lifting, verifies the physical hold,
and updates the planning world's grasp using the measured object pose.

The counter-to-table demo now runs a CRAM sequence:

```text
Query the deployed skill and the measured object state
Ground a bounded CRAM pick-up attempt
  Prepare the right arm
  Select and probe a station inside the trained region
  Navigate and re-check measured admission conditions
  Select an allowed, untried grasp reference
  RLPickUpAction(object_designator=bowl, arm=RIGHT)
  On a recoverable miss: verify, restore, and try another reference
Carry and place the bowl on the dining table
```

`GraspDemo.connect()` binds `AdaptivePickUpRuntime` through `RLPickUpContext`.
`GraspDemo.build_task_plan()` constructs the sequence and `perform_task_plan()`
executes it. The existing CLI, model, controller parameters and scene remain the
same. Navigation goals and grasp references are selected at execution time when
not specified on the command line. The household demo uses this plan for its bowl segment and then continues
with its existing mug/drawer sequence.

## Calling the action

After connecting the demo and registering its object, an individual action can
also be composed with other CRAM actions:

```python
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import execute_single
from garmi_cram_rl.pick_up_action import RLPickUpAction

action = RLPickUpAction(
    object_designator=demo.body,
    reference="bowl-r020",
    timeout=180.0,
)
with real_robot(collision_avoidance=True):
    execute_single(action, context=demo.context).perform()
```

Here `real_robot` means execution through the live ROS interface to Isaac Sim.
The robot must first be prepared and navigated into the policy's admission region.
Omit `reference` to use the existing rule that selects the first allowed reference
from the object's position mask. An explicitly requested reference is checked
against that mask at execution time. The action result records the reference
actually used, the request ID and physical grasp measurements.

## Execution and failure handling

- A deferred execution node separates Giskard motions from RL control. Expanding
  or displaying the plan does not run the policy.
- The bound runtime checks the object/arm binding and controller ownership. The
  existing bridge checks pose, posture, settling and grasp-reference admissibility.
- The bridge retains its exclusive ownership, heartbeat and cancellation protocol.
- Physical hold verification precedes the semantic attachment. The simulator
  continues to use physical contacts.
- A failed attempt raises `RLPickUpFailure`. The adaptive plan checks whether
  recovery is allowed, restores admissible starting conditions and tries another
  reference. Unsafe failures or exhaustion stop the enclosing plan before transport.
- Planning-world simulations cannot invoke the live controller. `no_execution`
  can expand the plan without sending commands.

The live viewer receives the action's start, success and failure states. Existing
navigation and placement subplans are mounted under the active task stage.

## Grounding and recovery

The simulator exposes a read-only `describe` query containing the deployed model
hash, supported object/arm, training region, reference bank, current allowed
references and measured admission status. The original pose and mask restrictions
are shared with the start check, so querying applicability cannot widen them.

CRAM's entity-query backend generates up to three `RLPickUpAttempt` candidates.
`RLPickUpAlternatives` extends its underspecified-action node for this live skill:
it defers decisions until execution and ends on the first success. Kinematic
worlds cannot simulate the learned physical grasp, so applicability and preparation
are checked explicitly instead of treating a kinematic grasp trial as evidence.

Station candidates align the base laterally with the object and prefer a shorter
navigation, using a 5 cm inset from the training rectangle. Up to six candidates
per attempt are checked in a disposable copy of the current CRAM world with
Giskard collision avoidance. The probe uses a Cartesian motion objective: stock
CRAM simulated navigation uses `SetOdometry`, which does not check the route.
Probe logs include solver steps, path length and final position error. Live
navigation is followed by another admission check against measured feedback.

Reference ordering comes from the frozen training bank. The current position mask
filters that list; already attempted references are excluded. This is a deterministic
rule, not a learned ranking or a guarantee that the grasp will succeed.

A retry is allowed only after `attempt_failed` or `timeout`, with the object still
supported, upright at the validated yaw/height, within the mask, stationary, free
of gripper contact and without forbidden robot contact. Recovery first probes
opening/parking in the planning world. The server independently rechecks the
conditions before releasing control. Giskard then opens the fingers and restores
the starting arm posture; navigation and reference selection run again against
fresh measurements. Collision, dropping, communication failure and unsupported
object poses stop the task. Placement is outside the grasp retry plan.

`--max-grasp-attempts` defaults to 3. `--base` and `--reference` can supply a
preferred first station or reference; omitting them enables automatic selection.
No model retraining or CRAM core source changes are required.

## Files and validation

- `garmi_cram_rl/pick_up_action.py`: action, execution boundary and task stages.
- `garmi_cram_rl/demo_runtime.py`: binding to the validated bowl controller.
- `garmi_cram_rl/adaptive_pick_up.py`: CRAM query grounding and bounded recovery.
- `garmi_cram_rl/skill_description.py` and `skill_selection.py`: applicability and candidate rules.
- `garmi_cram_rl/planning_probe.py`: collision-aware navigation/preparation probes.
- `cram_demo.py` and `web_demo.py`: task composition and live visualization.
- `tests/test_pick_up_action.py` and `tests/test_demo_runtime.py`: CPU tests using
  the actual CRAM plan executor with a test skill backend.

Run these tests in the CRAM environment after sourcing ROS, with `integration/`
on `PYTHONPATH`:

```bash
python -m pytest -q integration/tests/test_pick_up_action.py \
  integration/tests/test_demo_runtime.py integration/tests/test_web.py \
  integration/tests/test_adaptive_pick_up.py integration/tests/test_skill_selection.py \
  integration/tests/test_skill_description.py integration/tests/test_planning_probe.py
```

The native deployment source map includes the same action/runtime modules and
the corresponding demo overrides. Package validation uses:

```bash
python3 scripts/build_native_runtime.py verify
```


To run the two recorded, varied-position regression cases without specifying
stations or references, use the same validated workspace and Isaac Lab paths as
the household demo:

```bash
bash integration/validate_adaptive_pickup.sh
```

Outputs go to a new `runs/adaptive-pickup-*` directory. Each case saves a video,
physical success checks, selected station/reference, planning probes and recovery
events. CPU tests exercise clean misses and terminal failures using a controlled
test backend; these are distinguished from physical simulation outcomes.


## Decision information in the existing plan tree

`WebDemo` supplies a `DecisionBridge` to NeemHub. The existing Plan tab shows
readable stage names and attempt numbers, with route checks, station selection,
reference selection, hold verification and recovery outcomes as expandable
monitor details. Running, passed and rejected entries follow recorded execution
events. A rejected candidate remains visible after another candidate succeeds.

The presentation adapter preserves execution-node identity and status; its
monitor rows are display annotations, not extra robot actions. Candidate ordering
and path checks use the existing implementation. `decision-plan.json` stores the
latest displayed tree beside the task report. No CRAM core or viewer layout
changes are required. The same adapter is wired into the native deployment
override, including its household task through `WebDemo`.

For a local, read-only layout review, `preview_decision_tree.py` reconstructs
selected snapshots from a recorded task report and copies its video. Run it
with the pinned CRAM Python and `integration/` on `PYTHONPATH`:

```bash
python integration/preview_decision_tree.py \
  --report RUN/task/rl-transport-report.json \
  --video RUN/sim/replay.mp4 --out runs/decision-review
python3 -m http.server 8785 --bind 127.0.0.1 --directory runs/decision-review
```

This separate review page uses the same plan payload and expandable step layout.
Its snapshots are reconstructed from recorded events; they do not seek the video
or issue robot commands. It is not a new panel in the live Lab.

Local validation (2026-11-03): 32 tests passed across presentation, adaptive
execution, runtime binding, action lifecycle and the web server; package checks
and source hashes passed. Chromium rendered the local review page at
`runs/decision-tree-review-20261103/preview.png`. The preview uses the successful
`auto-left-near` recording from 2026-11-01. No new physics run or remote deployment
was performed for this presentation change.


## Local original-style interactive Lab

Start the existing NeemHub viewer and Jupyter Desktop side by side, with Isaac
Sim 5.1 running in the desktop. Use the validated workspace and IsaacLab paths:

```bash
export GARMI_BOWL_WORKSPACE=/home/zgao/FAME_RL
export GARMI_ISAACLAB=/home/zgao/FAME_RL/third_party/IsaacLab
bash integration/web/local-lab.sh
```

Open `http://127.0.0.1:8888/proxy/8781/live-lab.html` and choose **Run loop**.
The original 3D, Plan, Statechart, Knowledge and coordinate-frame views are
available. Plan is selected initially; EQL can be restored with the existing
**View** menu. **Stop** closes the current simulation and finalizes its recording.
Each successful round stays visible for five seconds, then a fresh simulator and
CRAM task restore the same initial scene and execute another round. The desktop
stays connected during simulator loading. The header shows the round number.
Stop cancels both the active round and any pending restart; a failed round stops
the loop for inspection. Each round keeps its own logs and plan. The local live mode skips background
MP4/JPEG capture to reduce rendering overhead; the Isaac Sim desktop and NeemHub
remain live. To also save video, launch with `GARMI_LAB_RECORD_VIDEO=1`.
Stopping the launcher closes its own services. All services bind locally.

The launcher stages code and saves outputs in `runs/garmi-local-live-*`, leaving
the validated asset/model sources read-only. It applies a small viewer adaptation
inside the disposable container: `layout=workspace` preserves the full interface
when the embedded viewer selects its live scene, uses a camera position inside
the apartment, and gives explicit action statuses precedence over aggregate
child statuses. Mapped actions' unexecuted primitive templates are omitted from
the display; they are not reported as executed motions. The CRAM execution tree,
policy and physical controller are unchanged.

Local live validation (2026-11-03): two full counter-to-table runs succeeded in
the original-style interactive Lab. The final run
`runs/garmi-local-live-20261103-124029/web/20261103-114728-624e` verified physical
placement with 4.9 mm horizontal error and completed the final arm parking.
The original Plan tab showed station checks, the selected reference and grasp
verification under `RLPickUpAction`; all executed action nodes reached SUCCEEDED.
Ten targeted presentation/web tests passed. Browser captures and
`live-validation.json` are saved in the parent `web/` directory. This validation preceded automatic looping. No remote publishing was performed.

Automatic-loop validation (2026-11-03): seven lifecycle/web tests passed. A real
round completed with verified physical placement, then automatically started a
second round. The bowl reset to the same position; initial base translation
differed by 0.47 mm after settling. Stop cancelled the second round and prevented
restart; a new loop was then started for local review. Evidence is saved in
`runs/garmi-local-live-20261103-131011/web/loop-validation.json` and
`reset-validation.json` in the same directory.

The simulator writes `sim/timing.jsonl` in local Lab runs, reporting real-time
factor and average costs of control, rendering, ROS, state publication and video
capture. Physics uses the same 0.005-second substeps and 0.05-second control
interval. Local performance settings do not alter the trained policy or Giskard
velocity limits.

In local live mode, Kit's main-loop rate limiter is disabled after environment
initialization, because the Python control loop already paces each 0.05-second
control step. `sim/render-settings.json` records the previous/new setting and
unchanged physics/control time steps. Viewport refresh cadence and image quality
are preserved. Recorded-video mode retains the original Kit settings.

Performance validation (2026-11-03): three 100-step windows after the initial
warm-up window measured real-time factors of 0.462 with video capture, 0.596
without capture, and 0.775 without capture plus Kit rate limiting. The last
configuration completed RL pick-up, physical transport, verified table placement
(7.2 mm horizontal error) and final parking. Automatic looping remains active.
Measurements and success evidence are in
`runs/garmi-local-live-20261103-140422/web/performance-comparison.json` and
`performance-validation.json`; actual speed varies with scene and system load.
