"""Run one isolated RoboKudo-to-CRAM-to-RL mug test inside the native image."""

import argparse
from pathlib import Path
import os
import signal
import subprocess
import time
import json

ROOT = Path(__file__).resolve().parents[2]
NATIVE = "/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python"
SOURCE = 'source /opt/ros/jazzy/setup.bash; source /home/jovyan/cram-vrb-lab/ros2_ws/install/setup.bash; exec "$@"'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--record", action="store_true")
    p.add_argument("--mug-map", nargs=2, type=float, default=[0.16, 7.26])
    p.add_argument(
        "--base-map", nargs=3, type=float, default=[0.0, 5.8, 1.5707963267948966]
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "ROS_DOMAIN_ID": "60",
        "GARMI_ENABLE_ROS": "1",
        "PYTHONUNBUFFERED": "1",
        "FASTRTPS_DEFAULT_PROFILES_FILE": str(ROOT / "integration/fastdds.xml"),
        "PYTHONPATH": ":".join(
            str(ROOT / x) for x in ["integration", "scripts", "cram-vrb-lab"]
        ),
        "GARMI_LIVE_PORT": "8767",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
    }
    processes = []

    def launch(name, command):
        with (a.out / (name + ".log")).open("w") as log:
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        processes.append(proc)
        return proc

    def wait_marker(proc, name, marker, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if marker in (a.out / (name + ".log")).read_text(errors="replace"):
                return
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited with {proc.returncode}")
            time.sleep(1)
        raise TimeoutError(f"{name} startup exceeded {seconds}s")

    try:
        sim = launch(
            "sim",
            [
                "bash",
                str(ROOT / "scripts/local_python.sh"),
                str(Path(__file__).with_name("sim.py")),
                "--out",
                str(a.out / "sim"),
                "--headless",
                "--max-seconds",
                "600",
                "--base-map",
                *map(str, a.base_map),
                "--mug-map",
                *map(str, a.mug_map),
                *(["--record"] if a.record else []),
            ],
        )
        wait_marker(sim, "sim", "ROBOKUDO_RL_READY", 240)
        giskard = launch(
            "giskard",
            [
                "bash",
                "-c",
                SOURCE,
                "native",
                NATIVE,
                str(ROOT / "integration/giskard_server.py"),
                "--robot",
                "garmi",
                "--scene",
                "garmi_apartment",
            ],
        )
        wait_marker(giskard, "giskard", "giskard is ready", 100)
        task = launch(
            "task",
            [
                "bash",
                "-c",
                SOURCE,
                "native",
                NATIVE,
                str(Path(__file__).with_name("task.py")),
                "--out",
                str(a.out / "task"),
            ],
        )
        code = task.wait(timeout=400)
        if code:
            raise RuntimeError(f"Task exited with {code}")
        (a.out / "run.json").write_text(
            json.dumps(
                dict(status="completed", mug_map=a.mug_map, base_map=a.base_map),
                indent=2,
            )
        )
    except BaseException as error:
        (a.out / "run.json").write_text(
            json.dumps(dict(status="failed", error=repr(error)), indent=2)
        )
        raise
    finally:
        for proc in reversed(processes):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


if __name__ == "__main__":
    main()
