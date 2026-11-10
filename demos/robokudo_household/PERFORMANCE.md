# Interactive household demo performance

The optimized RoboKudo household demo preserves the original robot, policies,
physics timestep, motion limits and contact history. It uses the existing
continuous bowl-to-table then mug-to-drawer task.

## Changes

- Always-updated, single-environment contact sensors acquire forces directly,
  avoiding a GPU index query on every physics tick. Isaac Lab still performs
  force acquisition and history updates at every 5 ms physics step.
- The head RGB-D camera renders only during the three perception captures.
  The camera resumes before a capture and pauses afterward; the observer
  viewport and remote desktop remain live.
- Desktop streaming uses a fixed 1280×720 connection, 25 fps target and
  `--no-window` to avoid the desktop window manager resizing the server frame.
- `sim/performance.jsonl` records wall time, simulated time and resident memory
  at approximately 2-second intervals.

These changes are local to `demos/robokudo_household`. Multi-environment
training and other task entry points retain their existing sensor updates.

## Benchmark

`benchmark.py` compares original, optimized contact, and optimized contact plus
idle camera configurations in the same initialized scene (100 control steps
per configuration). It checks exact force-buffer and timestamp equality and
retention of a one-tick contact pulse for the full 10-frame history.

| Configuration | Mean control-step time |
| --- | ---: |
| Original updates, active head camera | 154.9 ms |
| Optimized contact updates, active camera | 120.1 ms |
| Optimized contact updates, idle head camera | 98.1 ms |

The combined change reduces step time by 36.7% in this stationary-scene
microbenchmark. Full interactive task timings additionally include planning,
ROS communication, desktop streaming and task verification.

Raw benchmark evidence: `performance-validation/benchmark.json`.

Run the benchmark inside the Isaac runtime:

```bash
bash scripts/local_python.sh demos/robokudo_household/benchmark.py \
  --out /tmp/household-benchmark --headless \
  --experience /workspace/integration/garmi-desktop.kit
```

Use a fresh output directory for each run.

## Memory diagnosis

The old interactive configuration was killed by the host OOM killer. In an
instrumented reproduction, simulator RSS rose from roughly 9 GB to 35 GB in
about 20 seconds. Isolated camera movement and isolated policy execution both
remained at approximately 8.5 GB, and the isolated bowl grasp succeeded.

The streamed reproduction exposed an application-frame/client-resolution
mismatch. This preview disables dynamic stream resizing, requests HD on the
client, and launches Kit with `--no-window`, as recommended in the emitted
Isaac streaming warning. A 24 GiB simulator RSS ceiling stops a regressing
preview before exhausting this host. This is a demo runtime guard.

## Full interactive validation

Run `20261110-140834-2908` completed one continuous scene: perceived bowl pickup,
physical table placement, a fresh mug observation, mug pickup, drawer opening,
insertion and closure. The policy-input isolation and physical success checks
all passed. The drawer opened 37.1 cm; the bowl did not move during the mug task.
Simulator RSS stayed between 8.3 and 8.7 GiB, with no new OOM kill.

| Bowl segment | Earlier successful run | Reported slow run | Optimized run |
| --- | ---: | ---: | ---: |
| Place and retreat | 130.0 s | 154.3 s | 89.1 s |
| Arm parking | 63.5 s | 90.2 s | 40.8 s |
| Placement, verification and parking combined | 197.6 s | 248.6 s | 134.0 s |

Wall times vary between runs, including motion-planner trajectories and desktop
load. The separate stationary-scene benchmark controls the scene and compares
update costs directly. The interactive demo is faster but still slower than
real time. Full evidence: `performance-validation/household-run.json`.
