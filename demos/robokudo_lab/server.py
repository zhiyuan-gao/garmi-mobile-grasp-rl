"""Isolated local perception Lab; reuse the existing desktop and NeemHub services."""
import atexit
import functools
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from http.server import ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'integration'),str(ROOT/'demos/household_sequence')]
spec=importlib.util.spec_from_file_location('household_service',ROOT/'demos/household_sequence/server.py')
household=importlib.util.module_from_spec(spec);spec.loader.exec_module(household)
legacy=household.legacy
HERE=Path(__file__).resolve().parent
STAGES=['Initialize Isaac Sim','RoboKudo RGB-D localization','CRAM station selection and navigation','RL approach, close and lift','Verify physical hold','Completed']

class Controller(household.HouseholdController):
    def __init__(self):
        self.object_name='mug'
        super().__init__()
        self.meta={'status':'idle','message':'Ready: RoboKudo → CRAM → RL mug grasp.'}
        self.run=None
    def env(self):
        env=super().env();env.update(ROS_DOMAIN_ID='61',GARMI_PERCEPTION_OBJECT=self.object_name)
        if self.run:env['GARMI_PERCEPTION_RESULT']=str(self.run/'sim/perception/estimate.json')
        return env
    def start_viewer(self):
        old=self.services.pop('neemhub',None)
        if old:self.stop_process(old)
        env=self.env();env.pop('CRAMERA_SCENES',None)
        self.spawn('neemhub',self.native(HERE/'neemhub.py','8722'),service=True,env=env)
    def start(self):
        result=super().start()
        self.meta.update(reference='mug-v3-ref025' if self.object_name=='mug' else None,
                         scenario='robokudo_'+self.object_name)
        self.persist()
        return self.status()
    def simulation_command(self):
        position=[.162,7.262] if self.object_name=='mug' else [.2,7.4]
        return ['bash',str(ROOT/'scripts/local_python.sh'),str(ROOT/f'demos/robokudo_{self.object_name}/sim.py'),
            '--out',str(self.run/'sim'),'--desktop-stream','--headless','--record',
            '--experience',str(ROOT/'integration/garmi-desktop.kit'),'--max-seconds','7200',
            '--base-map','.08','5.9','1.5707963267948966',f'--{self.object_name}-map',*map(str,position)]
    def execute(self):
        try:
            self.start_viewer()
            env=self.env();env.update(GARMI_ENABLE_ROS='1',FASTRTPS_DEFAULT_PROFILES_FILE=str(ROOT/'integration/fastdds.xml'),DISPLAY=':1',LIVESTREAM='0',HEADLESS='1',ENABLE_CAMERAS='1')
            self.prepare_simulation(env)
            self.spawn('giskard',self.native(ROOT/'integration/giskard_server.py','--robot','garmi','--scene','garmi_apartment'))
            self.wait_ready('giskard',lambda:'giskard is ready' in legacy.tail(self.run/'giskard.log'),150)
            self.spawn('task',self.native(HERE/'live_task.py','--out',self.run/'task','--object',self.object_name))
            self.meta.update(status='running',message='Using the measured object position for CRAM and RL.',scenario='robokudo_'+self.object_name)
            self.persist();deadline=time.monotonic()+600
            while time.monotonic()<deadline:
                if self.stop_event.wait(.5):raise InterruptedError('Stopped by the user.')
                if any(self.processes[n].poll() is not None for n in ('sim','giskard','task')):
                    raise RuntimeError('A task process stopped; see run logs.')
                result=legacy.read_json(self.run/'task/result.json',{})
                if result.get('status')=='completed':
                    if not result['result']['success']:raise RuntimeError('Grasp verification failed')
                    self.meta.update(status='completed',message='Perception-guided grasp verified. Inspect RoboKudo ↗ or the CRAM plan ↗.',finished_at=time.time());self.persist();return
            raise TimeoutError('Task exceeded ten minutes')
        except BaseException as error:
            for name in ('task','giskard','streaming-client','sim'):
                proc=self.processes.pop(name,None)
                if proc:self.stop_process(proc)
            self.meta.update(status='stopped' if isinstance(error,InterruptedError) else 'failed',message=str(error),finished_at=time.time());self.persist()
    def status(self):
        data=super().status();stage=0
        estimate=legacy.read_json(self.run/'sim/perception/estimate.json',{}) if self.run else {}
        if self.run and (self.run/'sim/perception/rgb.png').exists():stage=1
        if estimate:stage=2
        state=data.get('state',{});events=data.get('events',[])
        if state.get('owner') in ('rl','holding'):stage=3
        if any(e['event']=='handover_hold_check' for e in events):stage=4
        if data['status']=='completed':stage=5
        data.update(task='robokudo',object=self.object_name,stage=stage,stages=STAGES,events=self.events[-60:],perception=estimate,
            perception_mode='Initial snapshot; fixed known orientation',perception_ready=bool(estimate))
        return data

class Handler(legacy.Handler):
    def do_GET(self):
        route=self.path.split('?')[0]
        if route=='/api/perception':
            p=self.control.run/'sim/perception/visualization.json' if self.control.run else None
            return self.json(legacy.read_json(p,{}) if p else {})
        if route=='/api/perception-image':
            return self.send_file(self.control.run/'sim/perception/rgb.png' if self.control.run else None)
        if route in ('/explorer.html','/explorer.js','/explorer.css','/workspace.js'):
            return self.send_file(HERE/route.lstrip('/'))
        if route=='/three.min.js':
            return self.send_file(Path('/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/cramera/src/cramera/web/vendor/three.min.js'))
        return super().do_GET()
    def do_POST(self):
        if self.path=='/api/object':
            if self.headers.get('X-Demo-Token')!=self.control.token:return self.json({'error':'Refresh the page'},403)
            if self.control.status()['active'] or self.control.processes:return self.json({'error':'Stop the current display first'},409)
            if int(self.headers.get('Content-Length',0))>128:return self.json({'error':'Invalid request'},400)
            data=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
            if data.get('object') not in ('mug','bowl'):return self.json({'error':'Unknown object'},400)
            self.control.object_name=data['object'];return self.json({'object':self.control.object_name})
        return super().do_POST()

if __name__=='__main__':
    control=Controller();atexit.register(control.close)
    server=ThreadingHTTPServer(('127.0.0.1',8782),functools.partial(Handler,controller=control))
    signal.signal(signal.SIGTERM,lambda *a:sys.exit(0))
    print('ROBOKUDO_LAB_READY',flush=True);server.serve_forever()
