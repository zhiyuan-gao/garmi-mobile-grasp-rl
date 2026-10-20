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
