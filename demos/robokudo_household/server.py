"""Existing continuous task service with a separate RoboKudo inspector."""
import atexit
import functools
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'integration'),str(ROOT/'demos/household_sequence')]
spec=importlib.util.spec_from_file_location('household_service',ROOT/'demos/household_sequence/server.py')
household=importlib.util.module_from_spec(spec);spec.loader.exec_module(household)
legacy=household.legacy

class Controller(household.HouseholdController):
    def env(self):
        env=super().env();env.update(ROS_DOMAIN_ID='63', ACCEPT_EULA='YES', OMNI_KIT_ACCEPT_EULA='YES', PRIVACY_CONSENT='YES')
        return env
    def start_viewer(self):
        old=self.services.pop('neemhub',None)
        if old:self.stop_process(old)
        env=self.env();env.pop('CRAMERA_SCENES',None)
        self.spawn('neemhub',self.native(HERE/'neemhub.py','8722'),service=True,env=env)
    def spawn(self,name,command,**kwargs):
        if name=='viewer-connect':
            command=['/opt/conda/bin/python',str(HERE/'connect_viewer.py')]
        if name=='task':
            command=self.native(HERE/'live_task.py','--output-dir',self.run/'task')
        return super().spawn(name,command,**kwargs)
    def simulation_command(self):
        return ['bash',str(ROOT/'scripts/local_python.sh'),str(HERE/'sim.py'),
            '--out',str(self.run/'sim'),'--assets',str(ROOT/'household/assets'),
            '--mug-workspace',str(ROOT/'household/mug'),'--desktop-stream','--headless','--kit_args=--no-window',
            '--experience',str(ROOT/'integration/garmi-desktop.kit'),'--max-seconds','2400']
    def observations(self):
        return legacy.read_json(self.run/'sim/perception/index.json',[]) if self.run else []
    def observation(self,identifier=None):
        values=self.observations()
        if not identifier and values:
            seq=legacy.read_json(self.run/'sim/sequence.json',{})
            stage=seq.get('events',[{}])[-1].get('stage','bowl_counter_to_table')
            identifier='bowl-initial' if stage=='bowl_counter_to_table' else ('mug-pregrasp' if any(v['id']=='mug-pregrasp' for v in values) else 'mug-initial')
        return next((v for v in values if v['id']==identifier),None)
    def status(self,identifier=None):
        s=super().status();obs=self.observation(identifier)
        estimate=legacy.read_json(self.run/obs['path']/'estimate.json',{}) if obs else {}
        s.update(task='robokudo_household',object=obs['object'] if obs else 'bowl',
            observations=self.observations(),perception=estimate,perception_ready=bool(estimate),
            perception_id=obs['id'] if obs else None,perception_path=obs['path'] if obs else None,
            events=self.events[-100:])
        # Record selection is independent of the task. During the second segment,
        # its grasp result is recorded by the original household simulator.
        sequence=legacy.read_json(self.run/'sim/sequence.json',{}) if self.run else {}
        s['sequence']=sequence
        return s

class Handler(legacy.Handler):
    def do_GET(self):
        url=urlsplit(self.path);route=url.path
        identifier=parse_qs(url.query).get('observation',[None])[0]
        if route=='/api/status':return self.json(self.control.status(identifier))
        if route in ('/api/perception','/api/perception-image'):
            obs=self.control.observation(identifier)
            folder=self.control.run/obs['path'] if obs else None
            if route=='/api/perception':return self.json(legacy.read_json(folder/'visualization.json',{}) if folder else {})
            return self.send_file(folder/'rgb.png' if folder else None)
        if route in ('/explorer.html','/explorer.js','/workspace.js'):
            return self.send_file(HERE/route.lstrip('/'))
        if route=='/explorer.css':return self.send_file(ROOT/'demos/robokudo_lab/explorer.css')
        if route=='/three.min.js':return self.send_file(Path('/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/cramera/src/cramera/web/vendor/three.min.js'))
        return super().do_GET()

if __name__=='__main__':
    control=Controller();control.meta.update(status='idle',message='Ready: RoboKudo bowl → table → RoboKudo mug → drawer.')
    atexit.register(control.close)
    server=ThreadingHTTPServer(('127.0.0.1',8782),functools.partial(Handler,controller=control))
    signal.signal(signal.SIGTERM,lambda *a:sys.exit(0))
    print('ROBOKUDO_HOUSEHOLD_READY',flush=True);server.serve_forever()
