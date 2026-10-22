"""Run the frozen actor in the original single-environment physics task.

Starts at the documented RL handover pose. This tests grasping only; it does
not test CRAM navigation or transport. Writes measured states and a verdict.
"""
import argparse
import json
from pathlib import Path
import time

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", default="01-center")
parser.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
cases = json.loads((ROOT / "integration/left_counter_episodes.json").read_text())
case = next(case for case in cases if case["id"] == args.case)
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import imageio.v2 as imageio
from garmi_cram_rl.contract import Bundle
from garmi_cram_rl.lab_backend import LabBackend

backend = None
writer = None
started = time.monotonic()
report = dict(case=case, test_scope="RL approach + fixed close/lift/hold",
              success=False, checkpoint="PPO update 7500 (frozen actor.ts)")
try:
    bundle = Bundle(ROOT / "results/cram-rl-integration/policy-7500", ROOT)
    backend = LabBackend(bundle, case["rl_start"], case["bowl"],
                         render=args.enable_cameras)
    if args.enable_cameras:
        bowl = backend.snapshot()["object_map_position"]
        backend.camera([p + d for p, d in zip(bowl, case["camera_offset"])],
                       [bowl[0], bowl[1], bowl[2] + .12])
        writer = imageio.get_writer(args.out / "replay.mp4", fps=10,
                                   macro_block_size=16)
    with (args.out / "states.jsonl").open("w", buffering=1) as trace:
        # Sample settled state for the original admission checks (0.5 s window).
        for _ in range(40):
            backend.step_classical(time.monotonic())
            backend.snapshot()
        if writer is not None:
            # Initialize the render product before recording; no physics steps.
            for _ in range(10):
                frame = backend.env.render()
            imageio.imwrite(args.out / "initial-frame.png", frame)
        report["initial"] = backend.snapshot()
        backend.admit(case["reference"])
        backend.begin(case["reference"])
        report["admitted"] = True
        reason = None
        max_steps = int(backend.env.cfg.episode_length_s / backend.env.step_dt) + 2
        for step in range(max_steps):
            reason = backend.step_policy()
            state = backend.snapshot()
            trace.write(json.dumps(dict(step=step, reason=reason, **state)) + "\n")
            if writer is not None and step % 2 == 0:
                frame = backend.env.render()
                writer.append_data(frame)
            if step % 100 == 0:
                print(json.dumps(dict(step=step, phase=state["phase"],
                                      measurements=state["grasp_measurements"])),
                      flush=True)
            if reason is not None:
                break
        backend.hold()
        report.update(success=reason == "success", reason=reason or "step_limit",
                      policy_steps=step + 1,
                      policy_simulation_seconds=(step + 1) * backend.env.step_dt,
                      final=backend.snapshot())
        if writer is not None:
            for _ in range(20):
                backend.step_classical(time.monotonic(), holding=True)
                if backend.control_step % 2 == 0:
                    frame = backend.env.render()
                    writer.append_data(frame)
            imageio.imwrite(args.out / "final-frame.png", frame)
except Exception as error:
    import traceback
    traceback.print_exc()
    report["error"] = repr(error)
    raise
finally:
    report["elapsed_seconds"] = time.monotonic() - started
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("GRASP_TEST_RESULT " + json.dumps(report), flush=True)
    if writer is not None:
        writer.close()
    if backend is not None:
        backend.close()
    app.close()

raise SystemExit(0 if report["success"] else 1)
