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
STAGES = ['启动仿真', 'CRAM 导航到柜台', 'RL 控制底盘和右臂接近', '固定程序闭爪、提起',
          'CRAM 搬运', 'CRAM 放置、检查、收臂']


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


def external_integration_running():
    # Match executable arguments, not a shell's text mentioning these files.
    names = {'/workspace/integration/sim.py', '/workspace/integration/giskard_server.py',
             '/workspace/integration/cram_demo.py', '/workspace/integration/web_demo.py'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            if names.intersection(a.decode(errors='replace') for a in args):
                return True
        except OSError:
            pass
    return False


class Controller:
    def __init__(self, services=True, external_jupyter=False):
        RESULTS.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.external_jupyter = external_jupyter
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.processes = {}
        self.services = {}
        self.worker = None
        self.events = []
        self.run = None
        self.meta = {'status': 'idle', 'message': '准备就绪。点击开始运行新的柜台到餐桌任务。'}
        previous = read_json(RESULTS / 'latest.json', {})
        if previous.get('run_id'):
            candidate = RESULTS / previous['run_id']
            if candidate.parent == RESULTS and candidate.is_dir():
                self.run = candidate
                self.meta = read_json(candidate / 'run.json', self.meta)
                if self.meta.get('status') in ('starting', 'running', 'stopping'):
                    self.meta.update(status='interrupted', message='上次页面服务退出，任务未确认完成。')
        if services:
            try:
                self.start_services()
            except BaseException:
                self.close()
                raise

    def env(self):
        return dict(os.environ, ROS_DOMAIN_ID='47', ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',
                    RMW_IMPLEMENTATION='rmw_fastrtps_cpp', PYTHONUNBUFFERED='1',
                    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    PYTHONPATH=f'{ROOT}/integration:{ROOT}/rl:{ROOT}/cram-vrb-lab:{ROOT}/scripts',
                    CRAMERA_DATA=str((self.run or RESULTS / 'viewer') / 'cramera'),
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
            raise RuntimeError('NeemHub 端口 8721 已被其他服务使用。')
        self.spawn('neemhub', self.native('-m', 'cramera.server', '8721', '--no-browser'), service=True)

    def start_services(self):
        # Dedicated local Jupyter entry. Never change an existing server/configuration.
        if not self.external_jupyter:
            if not port_free(8888):
                raise RuntimeError('端口 8888 已被占用；未修改已有 Jupyter 服务。')
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
                raise RuntimeError('已有任务在运行，请等待完成或停止本次运行。')
            ensure_training_idle(ROOT)
            if external_integration_running() or not port_free(8766):
                raise RuntimeError('已有仿真/规划进程或 live bridge 在运行；本页面未启动第二套。')
            run_id = time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(2)
            self.run = RESULTS / run_id
            self.run.mkdir()
            self.events = []
            self.stop_event.clear()
            self.meta = dict(run_id=run_id, status='starting', started_at=time.time(),
                             message='正在加载完整 GARMI 场景和策略…', stage_history=[],
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
                raise InterruptedError('已停止本次运行。')
            if self.processes[name].poll() is not None:
                raise RuntimeError(f'{name} 提前退出，请查看日志。')
            if predicate():
                return
        raise TimeoutError(f'{name} 启动超时，请查看日志。')

    def execute(self):
        try:
            self.start_viewer()
            env = self.env()
            env.update(FASTRTPS_DEFAULT_PROFILES_FILE=str(ROOT / 'integration/fastdds.xml'),
                       LD_LIBRARY_PATH='/usr/local/nvidia/lib64:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib')
            self.spawn('sim', ['/isaac-sim/python.sh', str(ROOT / 'integration/sim.py'),
                '--bundle', str(ROOT / 'results/cram-rl-integration/policy-7500'),
                '--out', str(self.run / 'sim'), '--full-scene', '--headless', '--enable_cameras',
                '--live-preview', '--max-seconds', '1800'], env=env)
            self.wait_ready('sim', lambda: 'CRAM_RL_READY' in tail(self.run / 'sim.log'), 240)
            self.spawn('giskard', self.native(ROOT / 'integration/giskard_server.py',
                       '--robot', 'garmi', '--scene', 'garmi_apartment'))
            # The task's normal service wait handles Giskard readiness.
            task = self.spawn('task', self.native(ROOT / 'integration/web_demo.py',
                              '--output-dir', self.run / 'task'))
            with self.lock:
                self.meta.update(status='running', message='仿真已启动，正在执行任务。')
                self.persist()
            deadline = time.monotonic() + 1500
            while task.poll() is None:
                if self.stop_event.wait(.5):
                    raise InterruptedError('已停止本次运行。')
                if time.monotonic() > deadline:
                    raise TimeoutError('任务超过 25 分钟，已停止。')
                if self.processes['sim'].poll() is not None or self.processes['giskard'].poll() is not None:
                    raise RuntimeError('仿真或规划服务退出，任务已停止。')
                self.status()  # Record observed phase changes even without a browser open.
            events = read_json(self.run / 'task/rl-transport-report.json', [])
            verified = any(e.get('event') == 'transport_result' and e.get('passed') for e in events)
            completed = any(e.get('event') == 'completed' for e in events)
            if task.returncode != 0 or not completed or not verified:
                error = next((e.get('error') for e in reversed(events) if e.get('event') == 'failed'), None)
                raise RuntimeError(error or '任务未通过完整放置检查，请查看日志。')
            outcome, message = 'completed', '碗已放到餐桌，物理放置检查和收臂均完成。'
        except InterruptedError as error:
            outcome, message = 'stopped', str(error)
        except BaseException as error:
            outcome, message = 'failed', str(error)
        finally:
            for name in ('task', 'giskard', 'sim'):
                process = self.processes.pop(name, None)
                if process:
                    self.stop_process(process)
        with self.lock:
            self.meta.update(status=outcome, message=message, finished_at=time.time())
            self.persist()

    def stop(self):
        with self.lock:
            if not self.worker or not self.worker.is_alive():
                raise RuntimeError('当前没有本页面启动的运行。')
            self.meta.update(status='stopping', message='正在停止本次任务并保存录像…')
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
            return self.json({'error': '请刷新页面后重试。'}, 403)
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
    args = parser.parse_args()
    if not port_free(args.port):
        raise RuntimeError(f'端口 {args.port} 已被占用。')
    controller = Controller(external_jupyter=args.external_jupyter)
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
