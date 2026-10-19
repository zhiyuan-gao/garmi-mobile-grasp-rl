# GARMI interactive demo

The container runs the CRAM + RL counter-to-table task in the complete GARMI apartment. Open the workspace and click **开始一次运行** to start the simulation. The default setup picks the bowl from the left counter using checkpoint 7500 and reference bowl-r020.

## Binder / VRB

Use this repository’s `vrb-demo` branch with the launch path `proxy/8781/`:

<https://binder.intel4coro.de/v2/gh/zhiyuan-gao/garmi-mobile-grasp-rl/vrb-demo?urlpath=proxy%2F8781%2F>

The Dockerfile extends the pinned CRAM digital twin image, installs Isaac Lab, verifies the packaged inference assets, and prepares the full apartment. The entrypoint starts the page service alongside the Jupyter command supplied by Binder. A GPU-backed session is required. See [deployment status](VRB_DEPLOYMENT.md) for the current VRB GPU allocation status.

## Local Docker

```bash
docker build -t garmi-vrb-demo -f binder/Dockerfile .
docker run --rm --runtime=nvidia --gpus all --shm-size=1g \
  -p 127.0.0.1:8891:8888 --name garmi-vrb-demo garmi-vrb-demo \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
  --IdentityProvider.token= --ServerApp.password=
```

Open <http://127.0.0.1:8891/proxy/8781/>. This local command exposes the session only on the host loopback address; Binder supplies authentication for remote sessions.

Runs save to `/workspace/results/cram-rl-web/<run-id>/`: a physical validation report, simulator states, process logs, Isaac video, and NeemHub recording. Copy results out before deleting the container if needed.

The runtime archive contains the frozen policy and its required assets; expert datasets and training runs are not part of the demo image. See [runtime provenance](../runtime/README.md).
