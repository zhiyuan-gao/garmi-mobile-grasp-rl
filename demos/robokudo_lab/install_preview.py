"""Install this preview in a disposable container; never run on the old Lab."""
import hashlib
import json
from pathlib import Path
import os
if os.environ.get('GARMI_PERCEPTION_PREVIEW')!='1':
    raise RuntimeError('Install only in the dedicated GARMI_PERCEPTION_PREVIEW container')
ROOT=Path(__file__).resolve().parents[2]
web=Path('/home/jovyan/cram-vrb-lab/demos/web_ui/index.html')
text=web.read_text()
# Match the saved Lab: embedded scene, full native inspector through its arrow.
text=text.replace('{ name: "NeemHub", path: "proxy/8722/?layout=workspace" }',
    '{ name: "NeemHub", path: "proxy/8722/?scene=__live__", openPath: "proxy/8722/?scene=__live__&layout=workspace" }')
text=text.replace('url: resolve(p.path), hideTopBar:',
    'url: resolve(p.path), openUrl: resolve(p.openPath || p.path), hideTopBar:')
text=text.replace('openBtn.href = pane.url;', 'openBtn.href = pane.openUrl || pane.url;')
tag='<script src="robokudo-workspace.js" defer></script>'
text=text.replace('<script src="/proxy/8782/workspace.js" defer></script>',tag)
web.with_name('robokudo-workspace.js').write_text(Path(__file__).with_name('workspace.js').read_text())
if tag not in text:text=text.replace('</head>',tag+'\n</head>')
web.write_text(text)
folder=Path.home()/'.config/autostart';folder.mkdir(parents=True,exist_ok=True)
for name in ['GARMI-RL-Control.desktop','GARMI-Household-Control.desktop']:
    p=folder/name
    if p.exists():p.write_text('[Desktop Entry]\nType=Application\nHidden=true\n')
(folder/'GARMI-RoboKudo-Control.desktop').write_text('[Desktop Entry]\nType=Application\nName=GARMI RoboKudo Preview\nExec=/opt/conda/bin/python '+str(ROOT/'demos/robokudo_lab/start_service.py')+'\nTerminal=false\nX-GNOME-Autostart-enabled=true\n')
# Snapshot used to acquire the bridge lock, then world-state locks, while the
# world callback could hold world-state locks and request the bridge lock.
# Gather the robot data before locking the bridge; the lock now only publishes
# already gathered payloads. No planner or motion controller is changed.
p=Path('/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/cramera/src/cramera/live/bridge.py')
s=p.read_text();before=s
old='''        with self._lock:
            self.transform_state = transforms
            self.sequence_number += 1
            self._robot_catalog = robot_models.catalog()
'''
new='''        robot_catalog = robot_models.catalog()
        model_bases = robot_models.root_poses()
        with self._lock:
            self.transform_state = transforms
            self.sequence_number += 1
            self._robot_catalog = robot_catalog
'''
if old in s:
    if s.count(old)!=1 or s.count('model_bases=robot_models.root_poses(),')!=1:raise RuntimeError('Unexpected NeemHub version')
    s=s.replace(old,new).replace('model_bases=robot_models.root_poses(),','model_bases=model_bases,')
    backup=p.with_suffix('.py.before-robokudo-preview')
    if not backup.exists():backup.write_text(before)
    p.write_text(s)
elif new not in s:raise RuntimeError('Unsupported NeemHub snapshot implementation')
original=p.with_suffix('.py.before-robokudo-preview').read_text()
report={'bridge_before':hashlib.sha256(original.encode()).hexdigest(),'bridge_after':hashlib.sha256(s.encode()).hexdigest(),'change':'Read robot state outside bridge publication lock'}
(ROOT/'results/cram-rl-web').mkdir(parents=True,exist_ok=True)
(ROOT/'results/cram-rl-web/installation.json').write_text(json.dumps(report,indent=2))
print('PERCEPTION_PREVIEW_INSTALLED')
