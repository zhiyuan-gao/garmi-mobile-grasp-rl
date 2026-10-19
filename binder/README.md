# GARMI interactive demo

Run a trained mobile grasping policy in the complete GARMI apartment: **CRAM navigation → RL base-and-right-arm approach → fixed close/lift → CRAM transport and placement on the dining table**.

The default task picks a bowl from the left counter using PPO checkpoint 7500 and reference `bowl-r020`. [Watch the recorded demo](https://zhiyuan-gao.github.io/franka-tabletop-tasks/demos/garmi/).

## What is included

- The exported policy (`actor.ts`), including its observation normalizer.
- All 40 fixed bowl grasp references, the position-based reference filter and the policy configuration.
- The adapted robot, bowl and kitchen assets, CRAM/RL integration and web interface.

The build automatically extracts and verifies [runtime/data.tar.gz](../runtime/data.tar.gz). The full apartment and CRAM/Isaac Sim installation come from the pinned upstream Lab image in [Dockerfile](Dockerfile). You do not need the author's FAME_RL workspace, expert datasets, a separate asset transfer or a training run to execute this demo.

## Run on a GPU workstation or server

Use a Linux host with Docker, NVIDIA Container Toolkit configured with the `nvidia` runtime, and an NVIDIA GPU/driver compatible with Isaac Sim 5.1.0, including Vulkan rendering. The host needs network access to pull the base image and install dependencies. The container supplies Isaac Sim, Isaac Lab and CRAM; their licenses and asset terms apply as described in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

Clone the **demo branch** and build from its repository root:

```bash
git clone --branch vrb-demo --single-branch https://github.com/zhiyuan-gao/garmi-mobile-grasp-rl.git
cd garmi-mobile-grasp-rl
docker build -t garmi-vrb-demo -f binder/Dockerfile .
```

Start the workspace:

```bash
docker run --rm --runtime=nvidia --gpus all --shm-size=1g \
  -p 127.0.0.1:8891:8888 --name garmi-vrb-demo garmi-vrb-demo \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
  --IdentityProvider.token= --ServerApp.password=
```

Open <http://127.0.0.1:8891/proxy/8781/> and click **开始一次运行** (Start a run). The page displays the live simulator image, task stages and the recorded replay. Opening the page alone does not start the simulator.

If Docker runs on a remote server, forward its local port from your own computer, replacing `your-user@your-gpu-server` with the server's SSH address, then open the same URL:

```bash
ssh -N -L 8891:127.0.0.1:8891 your-user@your-gpu-server
```

The Docker command binds the unauthenticated local Jupyter workspace to loopback. Binder deployments use JupyterHub authentication instead.

## Run on VRB / Binder

Build this repository's `vrb-demo` branch using `binder/Dockerfile`, with launch path **`proxy/8781/`**:

<https://binder.intel4coro.de/v2/gh/zhiyuan-gao/garmi-mobile-grasp-rl/vrb-demo?urlpath=proxy%2F8781%2F>

The entrypoint starts the demo web service alongside the Jupyter command supplied by Binder. The Lab session needs GPU access for both CUDA compute and Vulkan graphics. Building the image and opening the workspace do not allocate a GPU by themselves.

On 19 October 2026, the full task passed in a separate local GPU container. Our production Binder test opened the workspace successfully, but its session had no GPU/graphics driver access. See [deployment validation and platform setup](VRB_DEPLOYMENT.md) for that session's results and the remaining GPU configuration. Another Lab session's GPU availability must be checked separately.

For a colleague deploying the Lab, the handoff is: build the demo image, assign the GPU-backed session, open `proxy/8781/`, and run the task. Use this derived image for the RL demo; it packages the policy and integration that the original CRAM Lab image does not contain.

## Check the result and save it

A successful run ends with `status: completed` in `run.json`. Its `task/rl-transport-report.json` contains a `transport_result` event with `passed: true`, followed by a `completed` event. These checks include release, support, object stability and placement accuracy.

Each run is saved under `/workspace/results/cram-rl-web/<run-id>/`:

| File | Contents |
| --- | --- |
| `run.json` | Run status and timestamps |
| `task/rl-transport-report.json` | Task events and physical placement checks |
| `sim/replay.mp4` | Simulator recording |
| `sim/states.jsonl` | Recorded simulator states |
| `*.log` | Simulator, planner and task process output |

For local Docker, copy the results before stopping the `--rm` container:

```bash
docker cp garmi-vrb-demo:/workspace/results/cram-rl-web ./garmi-demo-results
```

For Binder, download the run directory through the workspace before ending the session. NeemHub recordings are saved with each run as well.

## If the page opens but the simulation fails

Check GPU access inside the running container:

```bash
docker exec garmi-vrb-demo nvidia-smi
docker exec garmi-vrb-demo python3 -c 'import ctypes, glob; print(glob.glob("/dev/nvidia*")); ctypes.CDLL("libcuda.so.1"); ctypes.CDLL("libGLX_nvidia.so.0")'
```

In a Binder terminal, run `nvidia-smi` and the `python3 -c ...` command directly. Missing devices or libraries need the host/VRB GPU configuration fixed. Passing these checks confirms basic device/driver access; the simulator log and a complete task run verify rendering and execution. The web service log is `/workspace/results/cram-rl-web/server.log`; task-specific logs are in the run directory.

## Record several left-counter setups

[left_counter_episodes.json](../integration/left_counter_episodes.json) defines four initial setups and one follow-up with different bowl locations, navigation starts and RL handover positions. All use checkpoint 7500 and reference `bowl-r020`. The grasp camera looks toward the bowl from the opposite side of the wrist and follows the episode's initial bowl position. In the recorded case 04, navigation stopped beyond the trained region and RL admission was rejected. Case 05 repeats that bowl position with the handover target moved 7 cm inward; the original attempt is retained.

With the workspace container running and no task active, record the cases sequentially:

```bash
docker exec garmi-vrb-demo python3 /workspace/integration/record_episodes.py \
  --cases /workspace/integration/left_counter_episodes.json \
  --out /workspace/results/left-counter-episodes
```

Each episode directory contains its initial conditions, success/failure summary, simulator video, states and task report. Use a new output directory for another batch. The web run button starts the default single case; the command above runs all five recorded conditions.

To make a review page with grasp clips and full-task previews, run the following where `ffmpeg` and `ffprobe` are installed, using the copied batch directory:

```bash
python3 integration/make_episode_report.py --runs /path/to/left-counter-episodes --out ./episode-review
```

Clip boundaries come from the recorded control owner and finishing phase. Failed cases remain in the review alongside successful ones.

## Training and publication

For collecting demonstrations or training a policy, use the [`main` branch instructions](https://github.com/zhiyuan-gao/garmi-mobile-grasp-rl/blob/main/docs/usage.md). That workflow has separate training asset/bootstrap preparation. The demo's `actor.ts` is an inference export; the full optimizer/value-network checkpoint and expert datasets are not included. See [runtime provenance](../runtime/README.md).

For the VRB overview, the [content preview](https://zhiyuan-gao.github.io/franka-tabletop-tasks/demos/garmi/vrb-overview.html) and [publication package](https://zhiyuan-gao.github.io/franka-tabletop-tasks/demos/garmi/vrb-intro-package.zip) contain videos, descriptions, project links and the Enter Lab handoff instructions.
