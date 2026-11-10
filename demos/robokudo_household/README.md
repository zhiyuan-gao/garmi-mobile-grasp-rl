# RoboKudo household demonstration

One continuous GARMI scene starts with a bowl and mug on the kitchen counter.
RoboKudo localizes both from the head RGB-D camera. CRAM selects the bowl approach
station and calls `RLPickUpAction`; the existing controller carries the bowl to the
dining table. The robot returns to the counter, captures a fresh mug observation,
and uses that position for the mug policy. The original left-hand drawer opening,
right-hand placement and left-hand closing sequence then completes the task.

The Lab keeps its NeemHub + Desktop layout. **RoboKudo ↗** opens the perception
inspector with a selector for `bowl-initial`, `mug-initial` and `mug-pregrasp`.
Each observation has its actual RGB image, support plane, target point cloud,
known-model fit and capture time. **Follow task** selects the relevant observation.
The NeemHub arrow opens the native Knowledge, Plan, Statechart, Kinematics and
Transforms views. The plan includes perception provenance from real run events.

## Local runtime

Build the native runtime using `scripts/build_native_runtime.py`, and unpack it
in a dedicated copy of the GARMI native image with the existing household assets.
Use Docker `--runtime=nvidia --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all` for
both CUDA and Vulkan access. Set `GARMI_PERCEPTION_HOUSEHOLD=1`, then run:

```bash
python /workspace/demos/robokudo_household/install_preview.py
```

Open `/files/demos/web_ui/index.html` through the container's authenticated
Jupyter server. **Start task** executes one complete sequence; the final scene
stays visible. **Stop** releases it. A subsequent start uses a fresh scene.

The default native household entry point and standalone perception demos retain
their existing code. This installer activates only the dedicated preview service.

## Data flow

- Detector input: rendered RGB-D, camera calibration, the known counter crop and
  known object meshes. Object orientation is fixed to the training orientation.
- Before each grasp, the policy observation and fixed finishing controller read
  the RoboKudo object pose. At lift, the estimate follows gripper kinematics.
- Each observation records an input-isolation check: policy construction must
  work with the physical object handle hidden, and change when the estimate moves.
- Physical success, collisions and contacts are measured by the simulator.
- Transport, placement and drawer operations retain their existing simulation
  feedback and controllers. The mug return station remains the original waypoint.
- The two objects remain in one physical scene throughout; switching the active
  policy does not reset the scene or respawn either object.

Runtime evidence is saved under `results/cram-rl-web/<run>/`: perception images
and point clouds in `sim/perception/`, policy input audit in `sim/policy-inputs.jsonl`,
physical sequence verification in `sim/sequence.json`, and the actual CRAM plan
in `task/decision-plan.json`.

Validate a finished run with:

```bash
python demos/robokudo_household/validate_run.py /path/to/run
```

This checks physical delivery, drawer closure, fresh mug perception after bowl
transport, the policy input audit and the simulator-object isolation checks.

Local runtime optimizations and measured timings are documented in
[PERFORMANCE.md](PERFORMANCE.md).
