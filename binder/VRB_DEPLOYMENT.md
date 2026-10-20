# VRB deployment status

The counter-to-table demo is packaged on the `vrb-demo` branch. The tested container revision is `b602e8d081275d19e507e19ec651fcc32b549c17`.

## Verified on 19 October 2026

- A separate local GPU container completed navigation, RL approach, physical close/lift, transport, placement and arm retraction. All physical placement checks passed; horizontal placement error was 0.0113 m. No object attachment was used.
- The isolated container passed 22 integration tests; the optional drawer test was skipped.
- Desktop/mobile workspace layout, embedded VS Code/JupyterLab, the run button, and seekable video playback passed browser checks.
- `binder.intel4coro.de` built and launched the image successfully. The remote workspace, embedded tools and run button passed browser checks.
- That remote session had no `/dev/nvidia*` devices, no `nvidia-smi`, and no loadable `libcuda.so.1` or `libGLX_nvidia.so.0`. Isaac Sim reported no Vulkan device. Remote simulation is therefore not yet operational.
- `binder.dev.intel4coro.de` returned HTTP 500 on `/health`; two build attempts returned `Unhandled error watching for build events`. The public launch uses the production endpoint instead.

## Additional local recordings on 19 October 2026

The grasp camera now looks from the opposite side of the wrist, relative to each bowl's initial position. The [episode gallery](https://zhiyuan-gao.github.io/franka-tabletop-tasks/demos/garmi/episodes/) contains four initial setups plus one follow-up, including every attempt:

- Cases 01, 02, 03 and 05 completed physical grasp, transport, placement and arm retraction. Placement errors were approximately 5.7, 5.5, 3.1 and 5.8 mm respectively.
- Case 04 stopped before RL began: navigation arrived at map Y = 6.524 m, outside the trained upper bound of 6.500 m. Case 05 kept the same bowl and navigation start, moved the handover goal from Y = 6.47 to 6.40 m, and passed. The original rejected attempt remains in the gallery.
- All cases use checkpoint 7500 and reference `bowl-r020`. The frozen policy bundle and its 37 training dependency hashes still match. These recordings ran in the local GPU development container; they do not establish remote VRB execution.

The case definitions and sequential recording command are included in the [quick start](README.md#record-several-left-counter-setups).

## Remaining platform setup

Provide this Lab session with an NVIDIA GPU that supports Isaac Sim 5.1 rendering, including compute and graphics driver access. The deployment image already sets `NVIDIA_DRIVER_CAPABILITIES=all`. An environment variable in the image alone cannot allocate a GPU.

The VRB infrastructure repository documents the relevant GPU and Vulkan settings:

- [BinderHub single-user GPU resource configuration](https://github.com/IntEL4CoRo/binder.intel4coro.de-deploy/blob/main/binder.yaml)
- [GPU sharing and graphics setup](https://github.com/IntEL4CoRo/binder.intel4coro.de-deploy/blob/main/docs/03-gpu-time-slicing.md)
- [GKE deployment configuration](https://github.com/IntEL4CoRo/binder.intel4coro.de-deploy/blob/main/binder-gke.yaml)

For example, the documented Helm configuration requests GPU resources with:

```yaml
jupyterhub:
  singleuser:
    extraResource:
      guarantees:
        nvidia.com/gpu: "1"
      limits:
        nvidia.com/gpu: "1"
    extraEnv:
      NVIDIA_DRIVER_CAPABILITIES: "all"
```

Apply the corresponding configuration to the GPU-backed environment assigned to this Lab, and ensure its NVIDIA runtime/driver injection and Vulkan ICD configuration follow the actual cluster setup. This repository does not change the VRB cluster configuration.

After GPU access is available, launch the pinned revision with `urlpath=proxy/8781/`, click the run button, and verify `transport_result.passed=true` followed by `completed` in the saved task report. Then remove the GPU-pending notice from the public demo page.

The existing VRB overview's **Enter Lab** link points to the gallery at <https://zhiyuan-gao.github.io/franka-tabletop-tasks/demos/>. The GARMI section is appended there, preserving the six existing demonstrations. Editing the overview page itself still requires its CMS editor or source repository.
