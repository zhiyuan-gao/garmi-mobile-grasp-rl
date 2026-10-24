# Online GARMI household task

The Lab runs one continuous task in Isaac Sim 6.1. The bowl and mug start on the
left kitchen counter. GARMI delivers the bowl to the dining table, returns for
the mug, opens the upper drawer with the left hand, places the mug inside with
the right hand, and closes the drawer.

The remote desktop starts `start_service.py` once when its graphical session is
created. It launches `server.py --autostart`, which owns one simulator, one
Giskard server, the CRAM task and NVIDIA's streaming viewer. The viewer connects
automatically. The final scene remains visible until Stop is pressed.

The original Workspace tabs and layout are retained without an outer task toolbar.
Inside Semantic Digital Twin Scene, the existing Task selector offers **Bowl and
Mug** in place of **garmi rl pick up**. This entry shows the actual live long task;
other recorded tasks remain available. Start and Stop are in the scene's existing
playback area. Selecting another task only changes the viewed scene; it does not
stop or restart the simulation. Reopening Bowl and Mug attaches to the existing
run; Start begins a new complete task only when explicitly pressed.


## Live synchronization

`live_task.py` keeps CRAM's world and NeemHub active across both object tasks.
Robot joint states and odometry come from the simulation bridge. Measured mug
poses and drawer opening update the corresponding twin objects. The same
measurements drive the stage display; no timer or recorded trajectory advances
it. Completion requires final physical verification and fresh synchronization
through the final stage.

Services use separate local ports: control 8782, NeemHub 8722, and live bridge
8767. Jupyter proxies these ports through its authenticated session. A scoped
file handler allows the packaged Workspace HTML to load its same-origin panes;
other Jupyter HTML files retain their normal sandbox.

## Runtime inputs

The source map packages the task and an isolated `household_rl/garmi_rl` module
tree. The bowl policy and shared resources are supplied by the existing frozen
runtime. `fetch_assets.py` installs the separately hashed household scene, mug
reference bank, profile and update-6500 policy. `unpack_assets.py` verifies the
archive and each member before extraction.

Outputs appear under `results/cram-rl-web/<run-id>/`: `run.json`,
`sim/sequence.json`, `sim/states.jsonl`, `task/rl-transport-report.json`,
`task/live-sync.jsonl` and process logs. `results/household-lab/server.log`
contains the control-service startup log.
