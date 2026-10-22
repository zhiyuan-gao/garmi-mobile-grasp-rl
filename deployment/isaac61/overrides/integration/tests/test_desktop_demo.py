"""Desktop launching preserves process ownership and rejects an existing GARMI run."""
import importlib.util
from pathlib import Path

import pytest

import web_server


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[2] / "scripts/run_desktop_demo.py"
    spec = importlib.util.spec_from_file_location("run_desktop_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detect_original_desktop_relative_command(tmp_path):
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(b"python\0demos/garmi_demo.py\0")
    (process / "cwd").symlink_to("/home/jovyan/cram-vrb-lab")
    assert web_server.external_integration_running(tmp_path) is True


def test_unrelated_script_with_same_filename_is_not_a_conflict(tmp_path):
    process = tmp_path / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(b"python\0demos/garmi_demo.py\0")
    (process / "cwd").symlink_to(tmp_path)
    assert web_server.external_integration_running(tmp_path) is False


def test_launcher_starts_one_run_and_closes_its_controller(launcher, tmp_path):
    class CompletedRun:
        run = tmp_path
        starts = 0
        closed = False

        def start(self):
            self.starts += 1
            return {"run_id": "test"}

        def status(self):
            return {"status": "completed", "stage": 0, "stages": ["done"],
                    "message": "completed", "active": False}

        def close(self):
            self.closed = True

    controller = CompletedRun()
    assert launcher.run(controller) == 0
    assert controller.starts == 1
    assert controller.closed is True


def test_interrupt_closes_owned_simulation(launcher, tmp_path):
    class InterruptedRun:
        run = tmp_path
        closed = False

        def start(self):
            return {"run_id": "test"}

        def status(self):
            raise KeyboardInterrupt()

        def close(self):
            self.closed = True

    controller = InterruptedRun()
    assert launcher.run(controller) == 130
    assert controller.closed is True


@pytest.mark.parametrize('failed_episode', [None, 3])
def test_ten_episodes_reuse_controller_and_continue_after_task_failure(
        launcher, tmp_path, failed_episode):
    controllers = []

    class Episode:
        closed = False
        simulation_failed = False

        def __init__(self):
            self.index = 0
            self.runs = []

        def start(self):
            assert not self.closed
            self.index += 1
            self.run = tmp_path / f'episode-{self.index}'
            self.runs.append(self.run)
            return {'run_id': self.run.name}

        def status(self):
            status = 'failed' if self.index == failed_episode else 'completed'
            return {'status': status, 'stage': 0, 'stages': ['done'],
                    'message': status, 'active': False}

        def close(self):
            self.closed = True

    def factory():
        controller = Episode()
        controllers.append(controller)
        return controller

    assert launcher.run_episodes(controller_factory=factory) == (1 if failed_episode else 0)
    assert len(controllers) == 1
    assert all(controller.closed for controller in controllers)
    assert len(set(controllers[0].runs)) == 10


def test_interrupt_stops_batch_before_next_episode(launcher, tmp_path):
    controllers = []

    class InterruptedEpisode:
        run = tmp_path
        closed = False

        def start(self):
            return {'run_id': 'interrupted'}

        def status(self):
            raise KeyboardInterrupt()

        def close(self):
            self.closed = True

    def factory():
        controller = InterruptedEpisode()
        controllers.append(controller)
        return controller

    assert launcher.run_episodes(controller_factory=factory) == 130
    assert len(controllers) == 1
    assert controllers[0].closed


def test_invalid_episode_count_rejected(launcher):
    for value in ['0', '-1']:
        with pytest.raises(launcher.argparse.ArgumentTypeError):
            launcher.positive_episode_count(value)


def test_failed_reset_stops_batch_and_closes_simulator(launcher, tmp_path):
    class BrokenSimulation:
        run = tmp_path
        simulation_failed = True
        starts = 0
        closed = False

        def start(self):
            self.starts += 1
            return {'run_id': 'broken'}

        def status(self):
            return {'status': 'failed', 'stage': 0, 'stages': ['reset'],
                    'message': 'reset failed', 'active': False}

        def close(self):
            self.closed = True

    controller = BrokenSimulation()
    with pytest.raises(RuntimeError, match='remaining episodes were cancelled'):
        launcher.run_episodes(controller_factory=lambda: controller)
    assert controller.starts == 1 and controller.closed


def test_only_owned_process_group_is_excluded(tmp_path, monkeypatch):
    for pid in (123, 124):
        process = tmp_path / str(pid)
        process.mkdir()
        (process / 'cmdline').write_bytes(
            b'python\0/home/jovyan/cram-vrb-lab/demos/sim.py\0')
    monkeypatch.setattr(web_server.os, 'getpgid', lambda pid: pid)
    assert web_server.external_integration_running(tmp_path, ignore_process_groups=(123,))
    (tmp_path / '124/cmdline').unlink()
    assert not web_server.external_integration_running(tmp_path, ignore_process_groups=(123,))


@pytest.mark.parametrize('reset_fails', [False, True])
def test_two_episodes_keep_simulator_and_stream_but_replace_planner(tmp_path, monkeypatch, reset_fails):
    import json
    from types import SimpleNamespace
    web = web_server
    monkeypatch.setattr(web, 'RESULTS', tmp_path)
    controller = web.Controller(services=False, desktop=True, keep_simulation=True)
    spawned, stopped = [], []
    monkeypatch.setattr(controller, 'start_viewer', lambda: None)
    monkeypatch.setattr(controller, 'wait_ready', lambda *args: None)

    def spawn(name, command, **kwargs):
        if name == 'sim':
            assert '--allow-episode-reset' in command
        if name == 'reset':
            assert 'task' not in controller.processes and 'giskard' not in controller.processes
            assert command[-1] == str(controller.run / 'sim')
            assert controller.processes['sim'].poll() is None
            (controller.run / 'reset.log').write_text('' if reset_fails else 'EPISODE_RESET_COMPLETE')
        if name == 'task':
            folder = controller.run / 'task'
            folder.mkdir()
            (folder / 'rl-transport-report.json').write_text(json.dumps([
                {'event': 'completed'}, {'event': 'transport_result', 'passed': True}]))
        code = (1 if reset_fails else 0) if name == 'reset' else 0 if name == 'task' else None
        process = SimpleNamespace(pid=len(spawned) + 100, returncode=code)
        process.poll = lambda: process.returncode
        controller.processes[name] = process
        spawned.append((name, process))
        return process

    def stop(process):
        stopped.append(process.pid)
        process.returncode = -15

    monkeypatch.setattr(controller, 'spawn', spawn)
    monkeypatch.setattr(controller, 'stop_process', stop)
    for episode in range(2):
        controller.run = tmp_path / str(episode)
        controller.run.mkdir()
        controller.execute()
        assert controller.meta['status'] == ('failed' if episode == 1 and reset_fails else 'completed')
    assert [name for name, _ in spawned].count('sim') == 1
    assert [name for name, _ in spawned].count('streaming-client') == 1
    assert [name for name, _ in spawned].count('giskard') == (1 if reset_fails else 2)
    sim = next(process for name, process in spawned if name == 'sim')
    assert (sim.pid in stopped) == reset_fails
    assert controller.simulation_failed == reset_fails
    controller.close()
    assert sorted(stopped) == sorted(process.pid for _, process in spawned)
    assert not controller.processes
