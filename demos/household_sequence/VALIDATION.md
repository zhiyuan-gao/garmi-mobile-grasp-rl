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

## Production Binder check (2026-10-24)

The candidate image built and was published successfully:
`intel4coro/zhiyuan-2dgao-2dcram-2dvrb-2dlab-000a82:75ee1c5e3b4a598c80ccca8bc840f19ff29e0673`.
The repository installer verified the packaged sources, frozen policy resources
and all 15 household asset files during the build.

[Candidate Lab entry](https://binder.intel4coro.de/v2/gh/zhiyuan-gao/cram-vrb-lab/75ee1c5e3b4a598c80ccca8bc840f19ff29e0673?urlpath=files%2Fdemos%2Fweb_ui%2Findex.html)

The production Workspace and remote desktop opened, selected the household task,
and automatically started the simulator launch. Actual simulation could not start:

- No `/dev/nvidia*` devices in the assigned session.
- `nvidia-smi` was absent.
- `libcuda.so.1` and `libGLX_nvidia.so.0` were not loadable.
- Isaac Sim reported `No device could be created` and `Failed to create any GPU devices`.
- The `.dev` Binder health endpoint returned HTTP 500.

The validation run was `20261024-144021-da83`; it was stopped after diagnosis.
Platform evidence was saved locally under
`/home/zgao/FAME_RL/deployment/household-platform-check-20261024/`.
The website Enter Lab link was not changed because remote execution has not passed.

### Remaining platform setup

The Lab needs a GPU-backed session with CUDA and Vulkan graphics driver access.
The platform repository documents this under
[`jupyterhub.singleuser.extraResource` and `extraEnv`](https://github.com/IntEL4CoRo/binder.intel4coro.de-deploy/blob/main/binder.yaml)
and [GPU graphics setup](https://github.com/IntEL4CoRo/binder.intel4coro.de-deploy/blob/main/docs/03-gpu-time-slicing.md).
Its published configuration targets the development Hub; the tested production
session does not expose these GPU resources. This is a platform allocation/runtime
configuration task, not a change to the CMS account or a repository environment variable.

Once GPU access is available, launch the pinned candidate above and verify:
`sim/sequence.json` has `status=completed`, `scene_initializations=1`, and a passing
`final_verification`; the control API reports completion with both live panes
connected. Then replace only the VRB article's Enter Lab field with the validated
production link, preserving all descriptions, videos and other task entries.
