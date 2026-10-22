"""Publish the successful RL recording beside the original GARMI NeemHub task."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile


def install(recording: Path, reference: Path, scenes: Path) -> Path:
    """Copy a complete recording using the original task's picker identity and style."""
    source = json.loads((recording / "scene.json").read_text())
    original = json.loads(reference.read_text())
    trajectory = json.loads((recording / source["trajectory"]).read_text())
    if not trajectory["frames"]:
        raise ValueError("The RL recording contains no playback frames")
    name = "garmi_rl_pick_up"
    destination = scenes / name
    if destination.exists():
        raise FileExistsError(destination)
    source.update(
        name=name,
        robotName=original["robotName"],
        environmentName=original["environmentName"],
        task="garmi rl pick up",
    )
    # Keep the recording's camera inside its apartment; use the native renderer.
    for field in ("rendering",):
        source.pop(field, None)
        if field in original:
            source[field] = original[field]
    scenes.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".garmi-rl-", dir=scenes) as staging:
        bundle = Path(staging) / name
        shutil.copytree(recording, bundle)
        (bundle / "scene.json").write_text(
            json.dumps(source, ensure_ascii=False, indent=1) + "\n")
        bundle.rename(destination)
    # NeemHub merges scene.json entries from its local scenes root on each request.
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path(
        "/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/"
        "cramera/scenes/garmi_pick_place/scene.json"))
    parser.add_argument("--scenes", type=Path,
                        default=Path("/home/jovyan/.cramera/scenes"))
    args = parser.parse_args()
    print(install(args.recording, args.reference, args.scenes))
