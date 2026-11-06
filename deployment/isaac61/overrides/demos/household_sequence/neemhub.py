"""Expose the live household task through NeemHub's existing scene picker."""
import argparse
from pathlib import Path
import socketserver

from cramera import paths, server

ROOT = Path(__file__).resolve().parent


def task_index():
    index = server.merged_scene_index()
    scenes = index.get('scenes', [])
    reference = next((s for s in scenes if s['name'] == 'garmi_rl_pick_up'),
                     {'robot': 'garmi', 'environment': 'GarmiApartment'})
    entry = dict(reference, name=paths.LIVE_SCENE_NAME, task='Bowl and Mug')
    return dict(index, default=paths.LIVE_SCENE_NAME,
                scenes=[s for s in scenes if s['name'] not in
                        ('garmi_rl_pick_up', paths.LIVE_SCENE_NAME)] + [entry])


def panel_source():
    text = (paths.WEB_ROOT / 'panels/robot_scene/panel.js').read_text()
    replacements = {
        'if (activeName === LIVE_SCENE_NAME || RecordingMode.isRecordingScene(activeName)) return;':
            'if (RecordingMode.isRecordingScene(activeName)) return;',
        "return fetch(sceneBase + 'scene.json').then(function (r) { return r.json(); }).then(loadScene);":
            'return HouseholdTask.loadScene(sceneBase, name).then(loadScene);',
    }
    for before, after in replacements.items():
        if text.count(before) != 1:
            raise RuntimeError('Unsupported NeemHub scene panel version')
        text = text.replace(before, after)
    marker = '    camera.position.set(c.x + 3.2, c.y + 1.6, c.z + 3.4);'
    if text.count(marker) != 1:
        raise RuntimeError('Unsupported NeemHub camera framing')
    return text.replace(marker, "    if (/[?&]layout=workspace(&|$)/.test(location.search)) camera.position.set(c.x + .9, c.y + 1.3, c.z + 1.8);\n    else " + marker.strip())


def workspace_source(route):
    text = (paths.WEB_ROOT / route.lstrip('/')).read_text()
    if route == '/core/scene.js':
        marker = '  function sceneOnly(framed) {\n'
        addition = "    if (/[?&]layout=workspace(&|$)/.test(location.search)) return false;\n"
    else:
        marker = '  function derivedStatus(item) {\n'
        addition = "    if (/[?&]layout=workspace(&|$)/.test(location.search) && item.n.derived === false && ['RUNNING','SUCCEEDED','FAILED','DONE'].includes(item.n.status)) return item.n.status;\n"
        if text.count('title: [label].concat(lines)') != 1:
            raise RuntimeError('Unsupported NeemHub plan payload')
        text = text.replace('title: [label].concat(lines)', 'derived: n.derived, title: [label].concat(lines)')
    if text.count(marker) != 1:
        raise RuntimeError('Unsupported NeemHub workspace renderer')
    return text.replace(marker, marker + addition)


class Handler(server.Handler):
    def send_text(self, text, content_type):
        data = text.encode()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = self.path.split('?')[0]
        if route == '/scenes/index.json':
            return self._send_json(task_index())
        if route in ('/', '/index.html'):
            text = (paths.WEB_ROOT / 'index.html').read_text()
            anchor = '<script src="config.js"></script>'
            if text.count(anchor) != 1:
                raise RuntimeError('Unsupported NeemHub index version')
            text = text.replace(anchor, '<link rel="stylesheet" href="household-task.css">\n'
                                '<script src="household-task.js"></script>\n' + anchor)
            return self.send_text(text, 'text/html; charset=utf-8')
        if route in ('/core/scene.js', '/panels/graph/panel.js'):
            return self.send_text(workspace_source(route), 'application/javascript')
        if route == '/panels/robot_scene/panel.js':
            return self.send_text(panel_source(), 'application/javascript')
        if route in ('/household-task.js', '/household-task.css'):
            return self._send_file(ROOT / route.lstrip('/'))
        return super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('port', type=int, default=8722, nargs='?')
    args = parser.parse_args()
    for route in ('/core/scene.js', '/panels/graph/panel.js'):
        workspace_source(route)
    panel_source()  # Fail clearly on an incompatible packaged frontend.
    socketserver.TCPServer.allow_reuse_address = True
    with server.LaboratoryServer(('127.0.0.1', args.port), Handler) as viewer:
        print(f'HOUSEHOLD_NEEMHUB_READY {args.port}', flush=True)
        viewer.serve_forever()


if __name__ == '__main__':
    main()
