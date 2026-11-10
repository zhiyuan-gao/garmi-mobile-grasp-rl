"""CRAM station reasoning and RLPickUpAction using the perception bridge."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "integration"),
    str(ROOT / "scripts"),
    str(ROOT / "cram-vrb-lab"),
]
from mug_task import MugDemo
from mobile_grasp import cancel_motion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    o = argparse.Namespace(
        reference="mug-v3-ref025",
        base=None,
        place_base=[0.9, 5.85, -0.85],
        target=[1.6, 5.32, 0.8],
        stage="rl-grasp",
        carry="physical",
        transport=False,
        resume_place=False,
        retreat=None,
        grasp_camera_offset=[-0.55, 0.28, 0.85],
        max_grasp_attempts=1,
    )
    demo = MugDemo(o, output=args.out)
    try:
        demo.run()
        state = demo.rl.snapshot()
        result = dict(
            status="completed", result=state["result"], snapshot=state["snapshot"]
        )
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        demo.record("completed", stage="robokudo_mug_grasp")
    except BaseException as error:
        if hasattr(demo, "context"):
            cancel_motion(demo.context)
        demo.record("failed", error=repr(error))
        (args.out / "result.json").write_text(
            json.dumps(dict(status="failed", error=repr(error)), indent=2)
        )
        raise
    finally:
        demo.close()


if __name__ == "__main__":
    main()
