"""Control boundary, phase attribution and seekable recordings of the web demo."""
import functools
import json
import threading
import urllib.error
import urllib.request

import pytest

import web_server as web


def test_actual_handover_phase():
    events = [{'event': 'navigation_arrival'}]
    assert web.phase(events, {'owner': 'rl', 'snapshot': {'phase': 0}}) == 2
    assert web.phase(events, {'owner': 'rl', 'snapshot': {'phase': 2}}) == 3
    events.append({'event': 'cram_control_returned'})
    # fixed_finish remains at hold after handback; this is now CRAM transport.
    assert web.phase(events, {'owner': 'cram', 'snapshot': {'phase': 4}}) == 4
    events.append({'event': 'placement_target'})
    assert web.phase(events, {'owner': 'cram', 'snapshot': {'phase': 4}}) == 5


def test_partial_state_write(tmp_path):
    path = tmp_path / 'states.jsonl'
    path.write_text('{"owner":"cram"}\n{"owner":"rl"}\n{"owner":')
    assert web.latest_state(path) == {'owner': 'rl'}


def test_control_auth_and_recording_ranges(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'RESULTS', tmp_path)
    controller = web.Controller(services=False)
    calls = []
    controller.start = lambda: calls.append('start') or {'status': 'starting'}
    run = tmp_path / 'run1'
    run.mkdir()
    (run / 'replay.mp4').write_bytes(b'0123456789')
    server = web.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(web.Handler, controller=controller))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(urllib.request.Request(base + '/api/start', method='POST'))
        assert failure.value.code == 403 and calls == []
        response = urllib.request.urlopen(urllib.request.Request(base + '/api/start', method='POST',
            headers={'X-Demo-Token': controller.token}))
        assert response.status == 202 and calls == ['start']
        response = urllib.request.urlopen(urllib.request.Request(base + '/runs/run1/replay.mp4',
            headers={'Range': 'bytes=3-5'}))
        assert response.status == 206 and response.read() == b'345'
        assert response.headers['Content-Range'] == 'bytes 3-5/10'
        with pytest.raises(urllib.error.HTTPError) as failure:
            urllib.request.urlopen(base + '/runs/%2e%2e/secret')
        assert failure.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_binder_uses_existing_jupyter_without_owning_it(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'RESULTS', tmp_path)
    calls = []
    monkeypatch.setattr(web.Controller, 'start_viewer', lambda self: calls.append('viewer'))
    monkeypatch.setattr(web.Controller, 'spawn', lambda *args, **kwargs: pytest.fail('Unexpected Jupyter child'))
    controller = web.Controller(external_jupyter=True)
    assert calls == ['viewer']
    assert 'jupyter' not in controller.services
    controller.close()


def test_desktop_launch_uses_local_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'RESULTS', tmp_path)
    controller = web.Controller(services=False, local_runtime=True, desktop=True)
    controller.run = tmp_path / 'run'
    command = controller.simulation_command()
    assert command[:2] == ['bash', str(web.ROOT / 'scripts/local_python.sh')]
    assert '--headless' in command
    assert '--desktop-stream' in command
    assert '--full-scene' in command
    assert '--enable_cameras' not in command
    assert '--live-preview' not in command
    position = command.index('--bowl-map')
    assert [float(value) for value in command[position + 1:position + 3]] == [.2, 7.4]


def test_headless_launch_remains_available(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'RESULTS', tmp_path)
    controller = web.Controller(services=False)
    controller.run = tmp_path / 'run'
    command = controller.simulation_command()
    assert command[0] == '/isaac-sim/python.sh'
    assert '--headless' in command
    assert '--enable_cameras' in command
    assert '--live-preview' in command
    assert '--desktop-stream' not in command


@pytest.mark.parametrize('desktop, port', [(True, 8765), (False, 8766)])
def test_live_bridge_and_scene_storage_match_viewer(tmp_path, monkeypatch, desktop, port):
    monkeypatch.setattr(web, 'RESULTS', tmp_path)
    controller = web.Controller(services=False, desktop=desktop)
    controller.run = tmp_path / 'episode'
    env = controller.env()
    assert env['GARMI_LIVE_PORT'] == str(port)
    assert env['CRAMERA_DATA'] == (str(web.Path('/home/jovyan/.cramera')) if desktop else
                                  str(controller.run / 'cramera'))
    assert env['GARMI_RECORDING_ARCHIVE'] == str(controller.run / 'cramera/scenes')


@pytest.mark.parametrize('desktop, port', [(True, 8765), (False, 8766)])
def test_busy_live_bridge_is_rejected_before_startup(tmp_path, monkeypatch, desktop, port):
    monkeypatch.setattr(web, 'RESULTS', tmp_path)
    monkeypatch.setattr(web, 'ensure_training_idle', lambda root: None)
    monkeypatch.setattr(web, 'external_integration_running', lambda **kwargs: False)
    checked = []
    def occupied(candidate):
        checked.append(candidate)
        return False
    monkeypatch.setattr(web, 'port_free', occupied)
    controller = web.Controller(services=False, desktop=desktop)
    with pytest.raises(RuntimeError, match='already running'):
        controller.start()
    assert checked == [port]
    assert controller.worker is None
