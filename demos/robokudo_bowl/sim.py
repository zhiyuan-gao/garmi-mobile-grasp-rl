"""Isolated full-apartment bowl perception and CRAM/RL grasp experiment."""

import argparse
from pathlib import Path
import sys
import json
import time
import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "integration"),
    str(ROOT / "rl"),
    str(ROOT / "scripts"),
    str(ROOT / "cram-vrb-lab"),
]
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", type=Path, required=True)
p.add_argument(
    "--base-map", nargs=3, type=float, default=[0.0, 5.8, 1.5707963267948966]
)
p.add_argument("--bowl-map", nargs=2, type=float, default=[0.2, 7.4])
p.add_argument("--capture-only", action="store_true")
p.add_argument("--record", action="store_true")
p.add_argument("--max-seconds", type=float, default=600.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
args.out.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
from backend import PerceivedBowlBackend
from garmi_cram_rl.contract import Bundle
from garmi_cram_rl.ros_bridge import Bridge
import rclpy

backend = node = recording = None
failed = False
try:
    bundle = Bundle(ROOT / "results/cram-rl-integration/policy-7500", ROOT)
    backend = PerceivedBowlBackend(
        bundle, args.base_map, args.bowl_map, args.out, args.capture_only
    )
    if not args.capture_only:
        rclpy.init()
        node = Bridge(backend, args.out / "events.jsonl")
        if args.record:
            from camera import ReviewCamera

            recording = ReviewCamera(backend, args.out / "replay.mp4")
        print("ROBOKUDO_RL_READY", flush=True)
        stopped = False

        def stop(*unused):
            global stopped
            stopped = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        deadline = time.monotonic() + args.max_seconds
        with (args.out / "states.jsonl").open("w", buffering=1) as trace:
            while app.is_running() and not stopped and time.monotonic() < deadline:
                started = time.monotonic()
                for _ in range(16):
                    rclpy.spin_once(node, timeout_sec=0.0)
                node.step()
                node.publish()
                if backend.control_step % 2 == 0:
                    if recording:
                        recording.append()
                    trace.write(
                        json.dumps(
                            dict(
                                **node.session.status(),
                                snapshot=backend.snapshot(),
                                fault=node.fault,
                            )
                        )
                        + "\n"
                    )
                if node.fault:
                    raise RuntimeError(node.fault)
                time.sleep(max(0.0, backend.env.step_dt - (time.monotonic() - started)))
except BaseException:
    failed = True
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if recording:
        recording.close()
    if node:
        node.destroy_node()
        rclpy.shutdown()
    if backend:
        backend.env.close()
    app.close(
        wait_for_replicator=False, skip_cleanup=True, exit_code=1 if failed else 0
    )
