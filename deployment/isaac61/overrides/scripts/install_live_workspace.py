"""Add live RL controls inside the existing NeemHub + Desktop tab."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = """  <section id="garmi-rl-controls" aria-label="GARMI RL controls">
    <div class="rl-row">
      <label for="garmi-demo-select">Demo</label>
      <select id="garmi-demo-select">
        <option value="garmi">GARMI</option>
        <option value="garmi-rl" selected>GARMI RL</option>
      </select>
      <span id="garmi-original-help" hidden>Original GARMI NeemHub. Start the demo using the GARMI_apartment icon on the Desktop.</span>
    </div>
    <div id="rl-runtime">
    <div class="rl-row"><strong>GARMI RL · CRAM + RL</strong>
      <button id="rl-start" disabled>Start RL Demo</button>
      <button id="rl-stop" disabled>Stop run</button>
      <span id="rl-status" role="status">Connecting to the RL service...</span>
      <a id="rl-details" target="_blank" rel="noopener">Camera / Run details ↗</a>
    </div>
    <ol id="rl-stages" aria-label="Task stages"></ol>
    <p id="rl-error" role="alert"></p>
    </div>
  </section>
"""


def install(workspace):
    web = workspace / "demos/web_ui"
    index = web / "index.html"
    original = index.read_text()
    updated = original
    if 'id="garmi-rl-controls"' not in original:
        replacements = {
            "  <main></main>": CONTROLS + "  <main></main>",
            "</head>": '  <link rel="stylesheet" href="garmi-rl-live/workspace-controls.css">\n</head>',
            "</body>": '  <script src="garmi-rl-live/workspace-controls.js"></script>\n</body>',
            "    function activate(i) {":
                '    function activate(i) {\n      document.getElementById("garmi-rl-controls").hidden = i !== 0;',
            "    activate(remembered >= 0 ? remembered : 0);":
                '    activate(window.location.hash === "#garmi-rl" ? 0 : remembered >= 0 ? remembered : 0);',
        }
        for before, after in replacements.items():
            if updated.count(before) != 1:
                raise RuntimeError(f"Cannot identify Workspace anchor: {before}")
            updated = updated.replace(before, after)
    # Upgrade an earlier live install without duplicating the controls or tabs.
    updated, count = re.subn(
        r'  <section id="garmi-rl-controls".*?</section>\n',
        lambda match: CONTROLS, updated, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError("Cannot identify the GARMI controls section")
    updated = updated.replace(
        '{ name: "NeemHub", path: "proxy/8721/?live=" + encodeURIComponent(BASE + "proxy/8766") }',
        '{ name: "NeemHub", path: "proxy/8711/" }')
    # An older reachability request must not overwrite the user's demo selection.
    updated = updated.replace('          iframe.src = url;',
                              '          if (!iframe.getAttribute("src")) iframe.src = url;')
    destination = web / "garmi-rl-live"
    destination.mkdir(exist_ok=True)
    for name in ("workspace-controls.css", "workspace-controls.js"):
        shutil.copy2(ROOT / "integration/web" / name, destination / name)
        digest = hashlib.sha256((destination / name).read_bytes()).hexdigest()[:12]
        updated = re.sub(
            rf'garmi-rl-live/{re.escape(name)}(?:\?v=[^"]*)?',
            f'garmi-rl-live/{name}?v={digest}', updated)
    backup = web / "index.html.before-garmi-rl-live"
    if not backup.exists():
        shutil.copy2(index, backup)
    index.write_text(updated)
    print(f"Installed live controls in NeemHub + Desktop: {index}")


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
    print(f"Configured RL control service autostart: {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/home/jovyan/cram-vrb-lab"))
    parser.add_argument("--desktop-home", type=Path,
                        help="Also configure desktop autostart under this home directory")
    args = parser.parse_args()
    install(args.workspace)
    if args.desktop_home:
        configure_autostart(args.desktop_home)
