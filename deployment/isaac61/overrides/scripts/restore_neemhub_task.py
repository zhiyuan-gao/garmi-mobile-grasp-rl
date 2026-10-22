"""Restore the saved GARMI RL task after a Binder container replacement.

Run this from binder/entrypoint.sh before starting cramera. The recording and
this script live on /mnt/dev-tools, outside the disposable container filesystem.
"""
import argparse
import json
from pathlib import Path

from install_neemhub_task import install

PROJECT = Path(__file__).resolve().parents[1]


def restore(workspace: Path, home: Path) -> Path:
    scenes = home / ".cramera/scenes"
    destination = scenes / "garmi_rl_pick_up"
    if destination.exists():
        scene = json.loads((destination / "scene.json").read_text())
        if scene.get("name") != destination.name or not (
            destination / scene["trajectory"]
        ).is_file():
            raise ValueError(f"Existing task is incomplete; refusing to overwrite {destination}")
        return destination
    return install(
        PROJECT / "results/cram-rl-web/20261020-164101-49d9/cramera/scenes/__recording__",
        workspace / "cognitive_robot_abstract_machine/cramera/scenes/garmi_pick_place/scene.json",
        scenes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path.home())
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    workspace = args.workspace or args.home / "cram-vrb-lab"
    print(restore(workspace, args.home))
