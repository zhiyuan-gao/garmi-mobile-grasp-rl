"""Run repeated counter-to-table episodes in one simulator, resetting between tasks."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integration"))

from web_server import Controller

DEFAULT_EPISODES = 10


def positive_episode_count(value):
    """Accept only a positive number of episodes from the command line."""
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("episodes must be at least 1")
    return count


def desktop_controller():
    """Own one simulator and streaming window for the entire batch."""
    return Controller(services=False, external_jupyter=True,
                      local_runtime=True, desktop=True, keep_simulation=True)


def ensure_workspace_alias():
    """Resolve the frozen asset paths without replacing another workspace."""
    alias = Path("/workspace")
    if not alias.exists() and not alias.is_symlink():
        alias.symlink_to(ROOT, target_is_directory=True)
    if alias.resolve() != ROOT:
        raise RuntimeError("/workspace must point to this project for the frozen assets")


def run(controller, *, close=True):
    """Start one task and keep its process ownership until completion or interruption."""
    try:
        metadata = controller.start()
        print(f"Run ID: {metadata['run_id']}", flush=True)
        print(f"Logs and scene traces: {controller.run}", flush=True)
        print("Display: WebRTC 640x360, matching the original GARMI demo. Video recording is disabled.", flush=True)
        print("Running one episode: navigation -> RL bowl grasp -> transport -> placement. Press Ctrl+C to stop.", flush=True)
        previous = None
        while True:
            status = controller.status()
            progress = (status["status"], status["stage"])
            if progress != previous:
                label = status["stages"][status["stage"]]
                print(f"[{time.strftime('%H:%M:%S')}] {label} — {status['message']}", flush=True)
                previous = progress
            if not status["active"]:
                return 0 if status["status"] == "completed" else 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping the RL demo and saving this run...", flush=True)
        return 130
    finally:
        if close:
            controller.close()


def run_episodes(episodes=DEFAULT_EPISODES, controller_factory=desktop_controller):
    """Reset the scene between episodes while retaining the simulator and stream."""
    succeeded = 0
    controller = controller_factory()
    try:
        for episode in range(1, episodes + 1):
            print(f"\n=== Episode {episode}/{episodes}: Resetting the robot, left-counter bowl and planning scene ===",
                  flush=True)
            code = run(controller, close=False)
            print(f"Episode {episode}/{episodes}: " +
                  ("Succeeded" if code == 0 else "Stopped" if code == 130 else "Failed"), flush=True)
            if code == 130:
                return code
            if controller.simulation_failed:
                raise RuntimeError('Simulator startup or reset failed; the remaining episodes were cancelled.')
            succeeded += code == 0
    finally:
        controller.close()
    print(f"\nFinished {episodes} episodes: {succeeded} succeeded, {episodes - succeeded} failed.",
          flush=True)
    return 0 if succeeded == episodes else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive_episode_count, default=DEFAULT_EPISODES,
                        help="Number of independently initialized episodes (default: 10)")
    parser.add_argument("--keep-open", action="store_true",
                        help="Keep the desktop terminal open after completion or an error")
    options = parser.parse_args()
    exit_signal = None

    def interrupt(signum, frame):
        nonlocal exit_signal
        exit_signal = signum
        # Let cleanup finish even if the window emits both HUP and TERM.
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(number, signal.SIG_IGN)
        raise KeyboardInterrupt()

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupt)
    folder = ROOT / "results/cram-rl-web"
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DISPLAY", ":1")
    code = 1
    try:
        with (folder / "desktop-launch.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("The GARMI RL desktop launcher is already running. Check the existing terminal.") from None
            ensure_workspace_alias()
            code = run_episodes(options.episodes)
    except (RuntimeError, OSError) as error:
        print(f"\nUnable to start GARMI RL: {error}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        code = 130
    if options.keep_open and exit_signal not in (signal.SIGHUP, signal.SIGTERM) and sys.stdin.isatty():
        try:
            input("\nPress Enter to close this window.")
        except (EOFError, OSError, KeyboardInterrupt):
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
