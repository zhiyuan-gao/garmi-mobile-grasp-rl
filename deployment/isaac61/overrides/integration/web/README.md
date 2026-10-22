# GARMI RL Workspace

## Desktop launcher

Double-click **GARMI RL Counter to Table** on the remote desktop to run **10 complete
counter-to-table episodes**. Isaac Sim, the loaded scene and the streaming window
stay open throughout the batch. Between episodes, the simulator resets the robot,
bowl, physics and RL control state; task and planning processes are recreated.
The existing bowl always starts on the left counter at map coordinates `(0.2, 7.4)`;
its position is fixed, not randomized.
The terminal shows the episode number, stage changes, results, and each
run's log directory. Press **Ctrl+C** to stop the batch; closing the launcher
terminal also requests cleanup of its simulation and planning processes.
After completion or an error, press Enter to close the terminal.

The launcher refuses a second desktop launch or an existing original GARMI
simulation. Stop the original demo in its terminal before starting RL.
The original GARMI launcher and Workspace page are unchanged.

The shortcut source is `integration/GARMI_RL_pick_up.desktop`, installed in
`/home/jovyan/Desktop` and `/home/jovyan/cram-vrb-lab/demos/desktop`.
Its command is:

```bash
/opt/conda/bin/python -u \
  /mnt/dev-tools/rl_project/garmi-mobile-grasp-rl/scripts/run_desktop_demo.py --episodes 10 --keep-open
```

Each episode runs navigation → RL approach and fixed close/lift → transport →
placement and physical verification. A failed task is reported and the next
episode starts after a scene reset; startup conflicts or a simulator/reset failure
stop the batch. The final
summary counts successes and failures. Use `--episodes 1` for a single run.
Each episode has independent outputs under `results/cram-rl-web/<run-id>/`,
including task logs, state traces and the recorded NeemHub scene. Running the batch does
not replace the saved NeemHub replay.
The simulator's console log is shared across the batch (`sim.log` in later runs
links to the first run); state traces and task recordings remain per episode.
Isaac Sim closes after all episodes finish or the batch is interrupted.

The desktop simulator uses the original GARMI display settings: a headless
640x360 WebRTC viewport, RaytracedLighting, and a target rendering cadence of
25 Hz. The standard Isaac Sim WebRTC client opens after the scene is ready;
connect to the local server using the same client controls as the original demo.
Camera video recording and JPEG previews are disabled for desktop runs to avoid
image readback and encoding overhead. The separate web camera mode still records
video. The RL physics step remains 5 ms with 10 substeps per policy/control step
(20 Hz); changing these to the original demo's 10 ms physics would change the
trained policy's dynamics. Actual rendering speed depends on simulation load.
Mouse pan, orbit and look speeds use Isaac Sim's standard defaults
in the live camera model after startup, so loaded preferences cannot override
the defaults in `integration/garmi-desktop.kit`.
These settings apply on the next simulator launch and are not saved to shared
Isaac Sim preferences.
The wheel also uses Kit's standard scale and movement speed. Scripted
camera changes update the zoom/orbit pivot in camera-local coordinates in the
USD session layer. This replaces Kit's stronger default pivot (hundreds of metres
away), which otherwise shadows a root-layer correction and amplifies pan and zoom.

## Native NeemHub task (current installation)

The additional outer demo bar has been removed. The initial Workspace view uses
the original NeemHub landing page, which automatically attaches when a demo is
running. Desktop RL uses the same live bridge port (8765) and local scene data
directory (`/home/jovyan/.cramera`) as original GARMI. Robot and object states
follow the running simulation; the two camera views remain independent. With
NeemHub's default auto-live setting, the next episode reconnects automatically.
The native task pickers are hidden while live, as in the original viewer; an
explicitly selected recorded scene remains a replay and does not auto-attach.
Each episode's recording is also archived in its own run directory.

In the existing **NeemHub + Desktop** tab, use the pickers under
**Semantic Digital Twin Scene**:

- Robot: **garmi**
- Environment: **GarmiApartment**
- Task: **garmi rl pick up**

The task uses the existing **Play NEEM**, timeline, layers, and scene panels,
alongside the original **pick up** task. This is a replay of the successful
CRAM → RL grasp → CRAM transport/placement run, using its measured trajectory
and original mesh bundle. The original GARMI task supplies the picker identity
and renderer defaults; the recording retains its camera inside the apartment.
No extra Workspace controls or simulation service
are required to play it.

Install the task into the original viewer's local scene catalog:

```bash
cd /mnt/dev-tools/rl_project/garmi-mobile-grasp-rl
python scripts/install_neemhub_task.py \
  --recording results/cram-rl-web/20261020-164101-49d9/cramera/scenes/__recording__
```

The built-in bundle is `cramera/scenes/garmi_rl_pick_up` within the CRAM stack.
The source recording and the existing `garmi_pick_place` bundle remain intact.
The installer refuses an existing destination rather than overwriting a saved
task. On 2026-10-20 the outer Workspace page and original GARMI desktop autostart
were restored, and the additional RL control service was stopped.

## Built-in Binder installation

The current installation serves `garmi_rl_pick_up` from the same shared
`cramera/scenes` directory as the original GARMI task. The temporary user-local
copy was moved to `~/.cramera/backups/garmi_rl_pick_up.before-native` after all
573 files were verified identical. Startup restore hooks have been removed.

The CRAM Lab Dockerfile now extracts the 24.6 MB recording archive during the
image build. The Workspace HTML is copied into the image and uses the native
NeemHub task pickers. This playback needs no persistent dev-tools mount at runtime.

The deployable source changes, recording archive, and checksums are retained on
persistent storage under `integration/native-binder/`. That package now also
includes the live RL source, frozen inference data, isolated Isaac Lab environment
installer, desktop entry, episode resets and camera fixes. Its build-time installer
uses image-local paths and needs no `/mnt/dev-tools` mount or startup recovery.
See its README for the prepared `yxzhan/cram-vrb-lab` deployment branch and the
normal Binder build/launch workflow. **The updated image has not yet been built
or deployed.** Recreating a pod from the previous image still produces the old
installation; publishing the source revision and successfully building it are
required to make RL available in new instances.

The old `restore_neemhub_task.py` remains a manual fallback for old images; it is
not used by the native image. `integration/neemhub-startup.patch` is superseded
by `integration/native-binder/workspace.patch`.

## Optional separate live controls (not installed in the current Workspace)

The existing Jupyter Workspace can host the live demo inside its
**NeemHub + Desktop** tab. A control bar starts/stops a run and displays measured
task stages. NeemHub connects to the run's live bridge; Desktop shows the
Isaac Sim window. **Camera / Run details** opens the camera, events, logs, and recording.
The other Workspace tabs remain available.

For this machine's local Isaac Sim 6.1 compatibility environment:

```bash
cd /mnt/dev-tools/rl_project/garmi-mobile-grasp-rl
python scripts/install_live_workspace.py --workspace /home/jovyan/cram-vrb-lab \
  --desktop-home /home/jovyan
python scripts/start_workspace_demo.py
```

Open the existing `files/demos/web_ui/index.html` URL and select
**NeemHub + Desktop**, or append `#garmi-rl`. Wait for the Desktop connection,
then click **Start RL Demo**. Merely opening the page does not start a simulation.
The **Demo** selector offers **GARMI** (the original NeemHub on port 8711) and
**GARMI RL**. Selection is remembered; `#garmi-rl` explicitly selects RL.
Switching views does not start or stop either simulation. Stop an active RL run
before launching the original GARMI from its Desktop icon.
The installer saves `index.html.before-garmi-rl-live` and can be rerun.
It upgrades older live controls and versions their CSS/JavaScript URLs to avoid
stale browser or proxy caches.
The page reuses Jupyter authentication and embeds no access token.

The control service listens on loopback port 8781, its NeemHub on 8721, and its
live bridge on 8766 (the separate web camera mode). The desktop launcher uses
the original Workspace viewer on 8711 and live bridge on 8765. It uses the project's local Python launcher for simulation
and the host's CRAM interpreter for planning, with ROS domain 47. The desktop
experience is `integration/garmi-desktop.kit`; the ROS libraries are enabled
only when the launcher receives `GARMI_ENABLE_ROS=1`.
Local runs use four times the original motion watchdog duration to accommodate
desktop and camera rendering on this machine. Timeout exceptions interrupt
blocked action waits, and the task's physical success checks are unchanged.
Live simulation uses the host GARMI demo's `RaytracedLighting` renderer.

The full apartment must be prepared before the first run:

```bash
/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python \
  integration/prepare_full_scene.py
```

The runtime requires `/workspace` to resolve to this project; the service
launcher creates that alias when absent and refuses a conflicting target.
Run files and the service log are under `results/cram-rl-web/`.
Only service-owned child process groups are stopped by the page.

This installation replaces the desktop session's automatic original GARMI
demo with `~/.config/autostart/GARMI-RL-Control.desktop`, which starts only
the control service. The old entry is disabled with `Hidden=true` and backed up
as `GARMI-Apartment-autostart.desktop.before-garmi-rl`. The previous demo's
process list is saved in `results/cram-rl-web/replaced-desktop-processes.json`.
Use `--desktop-home` to restore these autostart entries after a pod replacement.
The installer does not terminate an already running original simulation.

Recovery on 2026-10-20 restored the current `ox5vzu6u` instance. The earlier
`oy1trnwm` instance remained reachable. The retained run
`20261020-164101-49d9` completed all six stages, including physical transport,
release, support, and stationary placement verification (horizontal error
approximately 1.04 cm). Its full run was completed before recovery; recovery
checks cover the service, selector, browser navigation, and recording access.

## Separate workspace

A separate copy of the CRAM Workspace hosts the counter-to-table demo: CRAM navigation → RL base/right-arm approach → fixed close/lift → CRAM transport and placement.

The **GARMI RL** tab starts and stops runs, shows the live Isaac camera and actual task stages, and plays the recorded video. **NeemHub**, **VS Code**, and **JupyterLab** use the session’s existing proxy. Opening the page does not start a simulation.

For container and Binder launch instructions, see [binder/README.md](../../binder/README.md). The original CRAM Lab page remains unchanged.

Each run creates its own directory under `results/cram-rl-web/`. Success requires task completion and a passed physical placement check. Closing a run stops only that run’s processes.

## Recorded demo in the existing Workspace

To add a **GARMI RL** replay tab to the existing CRAM Workspace:

```bash
python scripts/install_workspace_demo.py --workspace /home/jovyan/cram-vrb-lab
```

This publishes the local Isaac Sim 6.1 grasp validation from
`results/isaac61-validation/08-center-video/` to
`demos/web_ui/garmi-rl/`, including its video, initial/final frames, report,
per-step states, and environment metadata. The page reads measurements directly
from the report. No simulation or additional server starts when viewing it.

Refresh `files/demos/web_ui/index.html` and select **GARMI RL**, or append
`#garmi-rl` to open that tab directly. Files use the existing Jupyter session;
no access token is embedded in the page. The installer preserves the existing
tabs, saves the original index as `index.html.before-garmi-rl`, and can be rerun
to republish the files without duplicating the tab.

This recording covers RL approach and fixed close/lift/hold only. The full
CRAM navigation, transport, and placement demo above is a separate workflow
and has not been validated by this local 6.1 recording.
