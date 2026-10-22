"""Local, separate CRAM Workspace entry for the GARMI RL counter-to-table demo.

Run inside fame-rl-dev. Only this server's child process groups are stopped.
"""
import argparse
import atexit
import functools
import json
import mimetypes
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from sim import ensure_training_idle

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'integration/web'
RESULTS = ROOT / 'results/cram-rl-web'
NATIVE = '/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python'
SOURCE = 'source /opt/ros/jazzy/setup.bash; source /home/jovyan/cram-vrb-lab/ros2_ws/install/setup.bash; exec "$@"'
STAGES = ['Start simulation', 'CRAM navigation to counter', 'RL base and right-arm approach', 'Scripted close and lift',
          'CRAM transport', 'CRAM placement, verification and arm parking']
LEFT_COUNTER_BOWL_MAP = (.2, 7.4)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def tail(path, size=16000):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def latest_state(path):
    for line in reversed(tail(path, 48000).splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            pass
    return {}


def phase(events, state):
    """Derive progress from actual task events and fixed-finish state, not a timer."""
    stage = 0
    for event in events:
        name, label = event.get('event'), event.get('label')
        if name == 'motion_start' and label == 'navigate_to_grasp_region':
            stage = max(stage, 1)
        if name == 'navigation_arrival':
            stage = max(stage, 2)
        if name in ('rl_grasp_result', 'cram_control_returned'):
            stage = max(stage, 4)
        if name == 'placement_target':
            stage = 5
    if state.get('owner') in ('rl', 'holding'):
        stage = max(stage, 3 if state.get('snapshot', {}).get('phase', 0) > 0 else 2)
    return stage


def port_free(port):
    with socket.socket() as sock:
        # Match HTTPServer's reuse setting: recently closed viewer connections
        # can otherwise be mistaken for an active listener after a run.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def external_integration_running(proc_root=Path('/proc'), ignore_process_groups=()):
    # Match executable arguments, not a shell's text mentioning these files.
    names = {str(base / 'integration' / name)
             for base in (ROOT, Path('/workspace'))
             for name in ('sim.py', 'giskard_server.py', 'cram_demo.py', 'web_demo.py')}
    names.update('/home/jovyan/cram-vrb-lab/demos/' + name
                 for name in ('sim.py', 'giskard_server.py', 'garmi_demo.py'))
    filenames = {Path(name).name for name in names}
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            if ignore_process_groups and os.getpgid(int(path.parent.name)) in ignore_process_groups:
                continue
            args = path.read_bytes().split(b'\0')
            for argument in args:
                candidate = Path(argument.decode(errors='replace'))
                if candidate.name not in filenames:
                    continue
                if not candidate.is_absolute():
                    candidate = (path.parent / 'cwd').resolve() / candidate
                if str(candidate.resolve()) in names:
                    return True
        except OSError:
            pass
    return False


class Controller:
    def __init__(self, services=True, external_jupyter=False, local_runtime=False, desktop=False,
                 keep_simulation=False):
        if keep_simulation and not desktop:
            raise ValueError('Persistent simulation requires desktop mode without video capture.')
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.external_jupyter = external_jupyter
        self.local_runtime = local_runtime
        self.desktop = desktop
        self.keep_simulation = keep_simulation
        self.simulation_log = None
        self.simulation_failed = False
        self.live_port = 8765 if desktop else 8766
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.processes = {}
        self.services = {}
        self.worker = None
        self.events = []
        self.run = None
        self.meta = {'status': 'idle', 'message': 'Ready. Start a new counter-to-table run.'}
        previous = read_json(RESULTS / 'latest.json', {})
        if previous.get('run_id'):
            candidate = RESULTS / previous['run_id']
            if candidate.parent == RESULTS and candidate.is_dir():
                self.run = candidate
                self.meta = read_json(candidate / 'run.json', self.meta)
                if self.meta.get('status') in ('starting', 'running', 'stopping'):
                    self.meta.update(status='interrupted', message='The previous control service exited before task completion was confirmed.')
        if services:
            try:
                self.start_services()
            except BaseException:
                self.close()
                raise

    def env(self):
        return dict(os.environ, ROS_DOMAIN_ID='47', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
                    GARMI_MOTION_TIMEOUT_SCALE='4' if self.local_runtime else '1',
                    RMW_IMPLEMENTATION='rmw_fastrtps_cpp', PYTHONUNBUFFERED='1',
                    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    PYTHONPATH=f'{ROOT}/integration:{ROOT}/rl:{ROOT}/cram-vrb-lab:{ROOT}/scripts',
                    GARMI_LIVE_PORT=str(self.live_port),
                    GARMI_RECORDING_ARCHIVE=str((self.run or RESULTS / 'viewer') / 'cramera/scenes'),
                    CRAMERA_DATA=(str(Path('/home/jovyan/.cramera')) if self.desktop else
                                  str((self.run or RESULTS / 'viewer') / 'cramera')),
                    CRAMERA_SCENES=str((self.run or RESULTS / 'viewer') / 'cramera/scenes'))

    def spawn(self, name, command, *, service=False, env=None):
        folder = RESULTS if service else self.run
        log = (folder / f'{name}.log').open('ab')
        try:
            process = subprocess.Popen(command, env=env or self.env(), cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()
        (self.services if service else self.processes)[name] = process
        return process

    def native(self, script, *arguments):
        return ['bash', '-c', SOURCE, 'web-demo', NATIVE, str(script), *map(str, arguments)]

    @staticmethod
    def stop_process(process):
        # The group id was allocated by start_new_session; no global pkill.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        # Also release descendants such as a static TF publisher.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def start_viewer(self):
        old = self.services.pop('neemhub', None)
        if old:
            self.stop_process(old)
        if not port_free(8721):
            raise RuntimeError('NeemHub port 8721 is already in use by another service.')
        self.spawn('neemhub', self.native('-m', 'cramera.server', '8721', '--no-browser'), service=True)

    def start_services(self):
        # Dedicated local Jupyter entry. Never change an existing server/configuration.
        if not self.external_jupyter:
            if not port_free(8888):
                raise RuntimeError('Port 8888 is already in use; the existing Jupyter service was left unchanged.')
            self.spawn('jupyter', ['/opt/conda/bin/jupyter', 'lab', '--allow-root', '--no-browser',
                       '--ip=127.0.0.1', '--port=8888', '--ServerApp.port_retries=0',
                       f'--ServerApp.root_dir={ROOT}', '--IdentityProvider.token=',
                       '--ServerApp.password='], service=True)
        self.start_viewer()

    def persist(self):
        if self.run:
            path = self.run / 'run.next.json'
            path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2) + '\n')
            path.replace(self.run / 'run.json')

    def start(self):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise RuntimeError('A task is already running. Wait for it to finish or stop the current run.')
            ensure_training_idle(ROOT)
            sim = self.processes.get('sim')
            if self.simulation_failed or (sim is not None and sim.poll() is not None):
                raise RuntimeError('The persistent simulator stopped or failed to reset. Start a new batch.')
            ignored = (sim.pid,) if self.keep_simulation and sim is not None else ()
            if external_integration_running(ignore_process_groups=ignored) or not port_free(self.live_port):
                raise RuntimeError('A GARMI simulation, planner or live bridge is already running. Press Ctrl+C in its terminal to stop it before starting the RL demo.')
            if self.desktop:
                display = os.environ.get('DISPLAY', ':1').split(':')[-1].split('.')[0]
                if not Path(f'/tmp/.X11-unix/X{display}').exists():
                    raise RuntimeError('Wait for the Desktop connection to complete before starting.')
            run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(2)
            self.run = RESULTS / run_id
            self.run.mkdir()
            self.events = []
            self.stop_event.clear()
            self.meta = dict(run_id=run_id, status='starting', started_at=time.time(),
                             message='Loading the full GARMI scene and policy...', stage_history=[],
                             reference='bowl-r020', scenario='counter-left-to-dining-table')
            self.persist()
            (RESULTS / 'latest.json').write_text(json.dumps({'run_id': run_id}) + '\n')
            self.worker = threading.Thread(target=self.execute, daemon=True)
            self.worker.start()
            return self.meta.copy()

    def wait_ready(self, name, predicate, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stop_event.wait(.3):
                raise InterruptedError('The run was stopped.')
            if self.processes[name].poll() is not None:
                raise RuntimeError(f'{name} exited early. Check the logs.')
            if predicate():
                return
        raise TimeoutError(f'{name} startup timed out. Check the logs.')

    def simulation_command(self):
        launcher = (['bash', str(ROOT / 'scripts/local_python.sh')] if self.local_runtime
                    else ['/isaac-sim/python.sh'])
        command = launcher + [str(ROOT / 'integration/sim.py'),
            '--bundle', str(ROOT / 'results/cram-rl-integration/policy-7500'),
            '--out', str(self.run / 'sim'), '--full-scene',
            '--bowl-map', *map(str, LEFT_COUNTER_BOWL_MAP),
            '--max-seconds', '1800']
        if not self.desktop:
            command.extend(['--headless', '--enable_cameras', '--live-preview'])
        else:
            command.extend(['--headless', '--desktop-stream'])
            if self.keep_simulation:
                command.append('--allow-episode-reset')
            if self.local_runtime:
                command.extend(['--experience', str(ROOT / 'integration/garmi-desktop.kit')])
        return command

    def prepare_simulation(self, env):
        """Load once, then reset the existing simulator before each subsequent task."""
        sim = self.processes.get('sim')
        if sim is None:
            self.spawn('sim', self.simulation_command(), env=env)
            self.simulation_log = self.run / 'sim.log'
            self.wait_ready('sim', lambda: 'CRAM_RL_READY' in tail(self.simulation_log), 240)
        else:
            if sim.poll() is not None:
                raise RuntimeError('The persistent simulator exited before the next episode.')
            (self.run / 'sim.log').symlink_to(self.simulation_log)
            reset = self.spawn('reset', self.native(ROOT / 'integration/reset_episode.py',
                                                   '--output', self.run / 'sim'))
            deadline = time.monotonic() + 60
            while reset.poll() is None:
                if self.stop_event.wait(.2):
                    raise InterruptedError('The run was stopped.')
                if sim.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('The simulator did not complete the episode reset.')
            if reset.returncode != 0 or 'EPISODE_RESET_COMPLETE' not in tail(self.run / 'reset.log'):
                raise RuntimeError('Episode reset failed. Check reset.log; the batch was stopped.')
            self.stop_process(self.processes.pop('reset'))
        if self.desktop:
            client = self.processes.get('streaming-client')
            if client is None or client.poll() is not None:
                if client is not None:
                    self.stop_process(client)
                self.spawn('streaming-client', ['isaacsim-webrtc-streaming-client', '--no-sandbox'], env=env)
        self.meta['simulator_pid'] = self.processes['sim'].pid

    def execute(self):
        simulation_ready = False
        try:
            self.start_viewer()
            env = self.env()
            env.update(GARMI_ENABLE_ROS='1',
                       FASTRTPS_DEFAULT_PROFILES_FILE=str(ROOT / 'integration/fastdds.xml'),
                       LD_LIBRARY_PATH='/usr/local/nvidia/lib64:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib')
            if self.desktop:
                env.update(DISPLAY=os.environ.get('DISPLAY', ':1'), LIVESTREAM='0',
                           HEADLESS='1', ENABLE_CAMERAS='0', ISAAC_HEADLESS='1', ISAAC_LIVESTREAM='1')
            self.prepare_simulation(env)
            simulation_ready = True
            self.spawn('giskard', self.native(ROOT / 'integration/giskard_server.py',
                       '--robot', 'garmi', '--scene', 'garmi_apartment'))
            # The task's normal service wait handles Giskard readiness.
            task = self.spawn('task', self.native(ROOT / 'integration/web_demo.py',
                              '--output-dir', self.run / 'task'))
            with self.lock:
                self.meta.update(status='running', message='Simulation started. Executing the task.')
                self.persist()
            deadline = time.monotonic() + 1500
            while task.poll() is None:
                if self.stop_event.wait(.5):
                    raise InterruptedError('The run was stopped.')
                if time.monotonic() > deadline:
                    raise TimeoutError('The task exceeded 25 minutes and was stopped.')
                if self.processes['sim'].poll() is not None or self.processes['giskard'].poll() is not None:
                    raise RuntimeError('The simulator or planner exited. The task was stopped.')
                self.status()  # Record observed phase changes even without a browser open.
            events = read_json(self.run / 'task/rl-transport-report.json', [])
            verified = any(e.get('event') == 'transport_result' and e.get('passed') for e in events)
            completed = any(e.get('event') == 'completed' for e in events)
            if task.returncode != 0 or not completed or not verified:
                error = next((e.get('error') for e in reversed(events) if e.get('event') == 'failed'), None)
                raise RuntimeError(error or 'The task did not pass all placement checks. Check the logs.')
            outcome, message = 'completed', 'The bowl is on the dining table. Physical placement checks and arm parking are complete.'
        except InterruptedError as error:
            outcome, message = 'stopped', str(error)
        except BaseException as error:
            outcome, message = 'failed', str(error)
        finally:
            sim = self.processes.get('sim')
            reusable = simulation_ready and sim is not None and sim.poll() is None
            self.simulation_failed = self.keep_simulation and not reusable
            names = ['task', 'giskard', 'reset']
            if not self.keep_simulation or not reusable or self.stop_event.is_set():
                names.extend(['streaming-client', 'sim'])
            for name in names:
                process = self.processes.pop(name, None)
                if process:
                    self.stop_process(process)
        with self.lock:
            self.meta.update(status=outcome, message=message, finished_at=time.time())
            self.persist()

    def stop(self):
        with self.lock:
            if not self.worker or not self.worker.is_alive():
                raise RuntimeError('There is no active run owned by this page.')
            self.meta.update(status='stopping', message='Stopping the task and saving the recording...')
            self.stop_event.set()
            self.persist()

    def status(self):
        with self.lock:
            result = self.meta.copy()
            state = {}
            if self.run:
                self.events = read_json(self.run / 'task/rl-transport-report.json', self.events)
                state = latest_state(self.run / 'sim/states.jsonl')
            stage = phase(self.events, state)
            history = self.meta.setdefault('stage_history', [])
            if self.worker and self.worker.is_alive() and (not history or history[-1]['stage'] != stage):
                history.append({'stage': stage, 'at': time.time()})
                self.persist()
            result.update(stage=stage, stages=STAGES, events=self.events[-20:], state=state,
                          active=bool(self.worker and self.worker.is_alive()),
                          preview=bool(self.run and (self.run / 'sim/preview.jpg').exists()),
                          replay=bool(self.run and (self.run / 'sim/replay.mp4').exists()
                                      and not (self.worker and self.worker.is_alive())),
                          recording=bool(self.run and (self.run / 'cramera/scenes/__recording__').exists()),
                          csrf_token=self.token)
            if self.run:
                result['files'] = ['run.json'] + [str(p.relative_to(self.run)) for p in
                    [self.run/'sim.log', self.run/'giskard.log', self.run/'task.log',
                     self.run/'task/rl-transport-report.json', self.run/'sim/replay.mp4'] if p.exists()]
            return result

    def close(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=50)
        for name in ('task', 'giskard', 'reset', 'streaming-client', 'sim'):
            process = self.processes.pop(name, None)
            if process:
                self.stop_process(process)
        for process in list(self.services.values()):
            self.stop_process(process)
        self.services.clear()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, controller, **kwargs):
        self.control = controller
        super().__init__(*args, directory=str(WEB), **kwargs)

    def log_message(self, fmt, *args):
        if self.command != 'GET' or not any(p in self.path for p in ('/api/status', '/api/preview')):
            super().log_message(fmt, *args)

    def json(self, value, code=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.headers.get('X-Demo-Token') != self.control.token:
            return self.json({'error': 'Refresh the page and try again.'}, 403)
        try:
            if self.path == '/api/start':
                return self.json(self.control.start(), 202)
            if self.path == '/api/stop':
                self.control.stop()
                return self.json({'status': 'stopping'}, 202)
            self.send_error(404)
        except (RuntimeError, OSError) as error:
            self.json({'error': str(error)}, 409)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == '/api/status':
            return self.json(self.control.status())
        if path == '/version.json':
            return self.json({'url': 'https://github.com/yxzhan/cram-vrb-lab/tree/41c1032'})
        if path == '/api/preview':
            file = self.control.run / 'sim/preview.jpg' if self.control.run else None
            return self.send_file(file)
        if path.startswith('/runs/'):
            file = (RESULTS / urllib.parse.unquote(path[6:])).resolve()
            if not file.is_relative_to(RESULTS.resolve()):
                return self.send_error(403)
            return self.send_file(file)
        return super().do_GET()

    def send_file(self, path):
        if path is None or not path.is_file():
            return self.send_error(404)
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        requested = self.headers.get('Range')
        if requested:
            try:
                unit, values = requested.split('=')
                lo, hi = values.split('-')
                if unit != 'bytes' or ',' in values:
                    raise ValueError()
                start = int(lo) if lo else max(0, size - int(hi))
                end = min(size - 1, int(hi)) if lo and hi else size - 1
                if not 0 <= start <= end < size:
                    raise ValueError()
                code = 206
            except ValueError:
                return self.send_error(416)
        self.send_response(code)
        self.send_header('Content-Type', mimetypes.guess_type(path)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-store')
        if code == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        try:
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(262144, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8781)
    parser.add_argument('--external-jupyter', action='store_true',
                        help='Reuse Binder/JupyterHub; do not start or stop its Jupyter server')
    parser.add_argument('--local-runtime', action='store_true',
                        help='Use the project local_python.sh simulation environment')
    parser.add_argument('--desktop', action='store_true',
                        help='Show the Isaac Sim window on the existing DISPLAY')
    args = parser.parse_args()
    if not port_free(args.port):
        raise RuntimeError(f'Port {args.port} is already in use.')
    controller = Controller(external_jupyter=args.external_jupyter,
                            local_runtime=args.local_runtime, desktop=args.desktop)
    atexit.register(controller.close)
    handler = functools.partial(Handler, controller=controller)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    def stop(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try:
        print(f'GARMI_RL_WEB_READY http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        controller.close()


if __name__ == '__main__':
    main()
