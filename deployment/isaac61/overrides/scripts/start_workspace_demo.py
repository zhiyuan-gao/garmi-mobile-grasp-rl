"""Start the RL page service once; simulations start only from the page button."""
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    alias = Path("/workspace")
    if not alias.exists() and not alias.is_symlink():
        alias.symlink_to(ROOT, target_is_directory=True)
    if alias.resolve() != ROOT:
        raise RuntimeError("/workspace must point to this project for the frozen assets")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8781/api/status", timeout=3) as response:
            if "csrf_token" in json.load(response):
                print("GARMI RL page service is already running.")
                return
    except (urllib.error.URLError, TimeoutError):
        pass
    folder = ROOT / "results/cram-rl-web"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "web-server.log").open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "integration/web_server.py"),
             "--external-jupyter", "--local-runtime", "--desktop"],
            cwd=ROOT, env=dict(os.environ, DISPLAY=os.environ.get("DISPLAY", ":1")),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (folder / "web-server.pid").write_text(str(process.pid) + "\n")
    print(f"Started GARMI RL page service (PID {process.pid}).")


if __name__ == "__main__":
    main()
