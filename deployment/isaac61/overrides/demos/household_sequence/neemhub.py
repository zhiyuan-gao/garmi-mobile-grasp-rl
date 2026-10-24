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
    return text


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
        if route == '/panels/robot_scene/panel.js':
            return self.send_text(panel_source(), 'application/javascript')
        if route in ('/household-task.js', '/household-task.css'):
            return self._send_file(ROOT / route.lstrip('/'))
        return super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('port', type=int, default=8722, nargs='?')
    args = parser.parse_args()
    panel_source()  # Fail clearly on an incompatible packaged frontend.
    socketserver.TCPServer.allow_reuse_address = True
    with server.LaboratoryServer(('127.0.0.1', args.port), Handler) as viewer:
        print(f'HOUSEHOLD_NEEMHUB_READY {args.port}', flush=True)
        viewer.serve_forever()


if __name__ == '__main__':
    main()
