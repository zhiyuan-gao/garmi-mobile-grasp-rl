"""Use NeemHub's native task picker without an outer Workspace control bar."""
import argparse
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
def install(workspace):
    web = workspace / "demos/web_ui"
    index = web / "index.html"
    original = index.read_text()
    updated = original.replace('const FALLBACK_BASE = "http://localhost:8888/";',
        'const FALLBACK_BASE = /^https?:$/.test(location.protocol) ? location.origin + "/" : "http://localhost:8888/";')
    # Restore the original Workspace chrome. Task selection belongs inside NeemHub.
    updated = re.sub(r'  <section id="garmi-rl-controls".*?</section>\n', '', updated, flags=re.DOTALL)
    updated = re.sub(r'  <(?:link|script)[^>]*(?:href|src)="garmi-rl-live/[^" ]+"[^>]*>(?:</script>)?\n', '', updated)
    updated = updated.replace('      document.getElementById("garmi-rl-controls").hidden = i !== 0;\n', '')
    for previous in (
        '{ name: "NeemHub", path: "proxy/8711/" }',
        '{ name: "NeemHub", path: "proxy/8721/?live=" + encodeURIComponent(BASE + "proxy/8766") }',
        '{ name: "NeemHub", path: "proxy/8722/?live=" + encodeURIComponent(BASE + "proxy/8767") }',
    ):
        updated = updated.replace(previous, '{ name: "NeemHub", path: "proxy/8722/" }')
    backup = web / "index.html.before-garmi-rl-live"
    if not backup.exists():
        shutil.copy2(index, backup)
    index.write_text(updated)
    print(f"Installed the native NeemHub Task entry: {index}")


def configure_autostart(home):
    folder = home / ".config/autostart"
    folder.mkdir(parents=True, exist_ok=True)
    old = folder / "GARMI-Apartment-autostart.desktop"
    if old.exists():
        backup = old.with_name(old.name + ".before-garmi-rl")
        if not backup.exists():
            shutil.copy2(old, backup)
        content = old.read_text()
        content = re.sub(r"(?m)^Hidden=.*\n?", "", content)
        old.write_text(content.rstrip() + "\nHidden=true\n")
    (folder / "GARMI-RL-Control.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=GARMI RL Control\n"
        f'Exec=/opt/conda/bin/python "{ROOT / "scripts/start_workspace_demo.py"}"\n'
        "Terminal=false\nX-GNOME-Autostart-enabled=true\n")
    (folder / "GARMI-Household-Control.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=GARMI Household Lab\n"
        f'Exec=/opt/conda/bin/python "{ROOT / "demos/household_sequence/start_service.py"}"\n'
        "Terminal=false\nX-GNOME-Autostart-enabled=true\n")
    print(f"Configured RL and household services: {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/home/jovyan/cram-vrb-lab"))
    parser.add_argument("--desktop-home", type=Path,
                        help="Also configure desktop autostart under this home directory")
    args = parser.parse_args()
    install(args.workspace)
    if args.desktop_home:
        configure_autostart(args.desktop_home)
