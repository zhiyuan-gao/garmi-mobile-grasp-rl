"""Publish the recorded grasp demo into an existing CRAM Workspace files page."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TAB = '      { panes: [{ name: "GARMI RL", path: "files/demos/web_ui/garmi-rl/index.html" }] },'
ANCHOR = '      { panes: [{ name: "LeRobot Data", path: "proxy/absolute/3000/" }] },'
OLD_ACTIVATION = "    activate(remembered >= 0 ? remembered : 0);"
NEW_ACTIVATION = """    const requested = window.location.hash === "#garmi-rl"
      ? targets.findIndex(t => t.name === "GARMI RL") : -1;
    activate(requested >= 0 ? requested : remembered >= 0 ? remembered : 0);"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/home/jovyan/cram-vrb-lab"))
    parser.add_argument("--run", type=Path,
                        default=ROOT / "results/isaac61-validation/08-center-video")
    args = parser.parse_args()
    web = args.workspace / "demos/web_ui"
    index = web / "index.html"
    original = index.read_text()
    updated = original
    if TAB not in updated:
        if updated.count(ANCHOR) != 1:
            raise RuntimeError("Cannot locate Workspace tab list; no files changed.")
        updated = updated.replace(ANCHOR, ANCHOR + "\n" + TAB)
    if NEW_ACTIVATION not in updated:
        if updated.count(OLD_ACTIVATION) != 1:
            raise RuntimeError("Cannot locate Workspace tab activation; no files changed.")
        updated = updated.replace(OLD_ACTIVATION, NEW_ACTIVATION)

    sources = {name: args.run / name for name in (
        "replay.mp4", "initial-frame.png", "final-frame.png", "report.json", "states.jsonl")}
    sources["environment.json"] = args.run.parent / "environment.json"
    sources["index.html"] = ROOT / "integration/web/replay.html"
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    # Reject malformed metadata before modifying the served page.
    for name in ("report.json", "environment.json"):
        json.loads(sources[name].read_text())
    destination = web / "garmi-rl"
    destination.mkdir(exist_ok=True)
    for name, source in sources.items():
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            shutil.copyfile(source, temporary)
            temporary.chmod(0o644)
            temporary.replace(destination / name)
        finally:
            temporary.unlink(missing_ok=True)
    if updated != original:
        backup = web / "index.html.before-garmi-rl"
        if not backup.exists():
            shutil.copy2(index, backup)
        with tempfile.NamedTemporaryFile(mode="w", dir=web, delete=False) as tmp:
            tmp.write(updated)
            temporary = Path(tmp.name)
        temporary.chmod(index.stat().st_mode & 0o777)
        temporary.replace(index)
    print(f"Published GARMI RL replay to {destination}")
    print("Open the existing Workspace and select GARMI RL, or append #garmi-rl.")


if __name__ == "__main__":
    main()
