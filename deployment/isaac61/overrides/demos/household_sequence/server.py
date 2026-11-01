"""One online household task, with a persistent Desktop and live NeemHub twin."""
import argparse
import atexit
import functools
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from http.server import ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'integration'))
import web_server as legacy
STAGES=legacy.STAGES+['Return to counter','RL mug approach and fixed close/lift','Carry mug to drawer',
    'Open drawer with left hand','Place mug inside','Close drawer with left hand','Completed']
EXTRA={'return_to_counter_mug':6,'mug_rl_approach':7,'carry_mug_to_drawer':8,'open_drawer_left':9,
       'place_mug_inside':10,'close_drawer_left':11,'completed':12}

class HouseholdController(legacy.Controller):
    def __init__(self, **kwargs):
        super().__init__(services=False,external_jupyter=True,local_runtime=True,desktop=True)
        self.live_port=8767
        self.meta={'status':'idle','message':'Ready for one counter → table → drawer task.'}
        self.run=None
        self.start_viewer()

    def env(self):
        env=super().env();env['ROS_DOMAIN_ID']='49';return env

    def start_viewer(self):
        old=self.services.pop('neemhub',None)
        if old:self.stop_process(old)
        if not legacy.port_free(8722):raise RuntimeError('Household viewer port 8722 is occupied')
        viewer_env=self.env()
        # Show the shared native Task catalog, while CRAMERA_DATA still supplies
        # the live world. The simulator retains its per-run recording directory.
        viewer_env.pop('CRAMERA_SCENES',None)
        self.spawn('neemhub',self.native(ROOT/'demos/household_sequence/neemhub.py','8722'),service=True,env=viewer_env)

    def spawn(self,name,command,**kwargs):
        if name=='streaming-client':
            command=command+['--remote-debugging-port=9231','--remote-debugging-address=127.0.0.1']
        return super().spawn(name,command,**kwargs)

    def prepare_simulation(self,env):
        super().prepare_simulation(env)
        viewer=self.spawn('viewer-connect',['/opt/conda/bin/python',str(ROOT/'demos/household_sequence/connect_viewer.py')])
        try:
            if viewer.wait(timeout=100)!=0:raise RuntimeError('Desktop viewer connection failed; see viewer-connect.log')
        finally:
            self.stop_process(self.processes.pop('viewer-connect'))

    def wait_ready(self,name,predicate,seconds):
        return super().wait_ready(name,predicate,900 if name=='sim' else seconds)

    def simulation_command(self):
        return ['bash',str(ROOT/'scripts/local_python.sh'),str(ROOT/'demos/household_sequence/sim.py'),
                '--out',str(self.run/'sim'),'--assets',str(ROOT/'household/assets'),
                '--mug-workspace',str(ROOT/'household/mug'),'--desktop-stream','--headless',
                '--experience',str(ROOT/'integration/garmi-desktop.kit'),'--max-seconds','2400']

    def start(self):
        # Completed displays keep their processes until an explicit new run/Stop.
        if not (self.worker and self.worker.is_alive()):
            for name in ('task','giskard','streaming-client','sim'):
                process=self.processes.pop(name,None)
                if process:self.stop_process(process)
        if not legacy.port_free(8765) or not legacy.port_free(8766):
            raise RuntimeError('Another GARMI demo is running; stop it before starting the household task.')
        return super().start()

    def execute(self):
        try:
            self.start_viewer()
            env=self.env();env.update(GARMI_ENABLE_ROS='1',FASTRTPS_DEFAULT_PROFILES_FILE=str(ROOT/'integration/fastdds.xml'),
                DISPLAY=os.environ.get('DISPLAY',':1'),LIVESTREAM='0',HEADLESS='1',ENABLE_CAMERAS='0',ISAAC_HEADLESS='1',ISAAC_LIVESTREAM='1')
            self.prepare_simulation(env)
            self.spawn('giskard',self.native(ROOT/'integration/giskard_server.py','--robot','garmi','--scene','garmi_apartment'))
            task=self.spawn('task',self.native(ROOT/'demos/household_sequence/live_task.py','--output-dir',self.run/'task'))
            self.meta.update(status='running',message='Running one continuous household task.',scenario='bowl-counter-table_then_mug-counter-drawer')
            self.persist();deadline=time.monotonic()+2400
            while time.monotonic()<deadline:
                if self.stop_event.wait(.5):raise InterruptedError('Stopped by the user.')
                sequence=legacy.read_json(self.run/'sim/sequence.json',{})
                if sequence.get('status')=='failed':raise RuntimeError(sequence.get('error','Simulation failed'))
                if any(self.processes[name].poll() is not None for name in ('sim','giskard','task')):
                    raise RuntimeError('A task process exited; see the run logs.')
                self.status()
                sync_error=next((e.get('error') for e in self.events if e.get('event')=='live_sync_failed'),None)
                if sync_error: raise RuntimeError('NeemHub synchronization failed: '+sync_error)
                if sequence.get('status')=='completed':
                    if sequence.get('scene_initializations')!=1 or not any(e.get('event')=='final_verification' and e.get('passed') is True for e in sequence.get('events',[])):
                        raise RuntimeError('Missing physical final verification')
                    sync=legacy.latest_state(self.run/'task/live-sync.jsonl')
                    if sync.get('simulation_seconds',-100) < sequence['events'][-1]['simulation_seconds']-2:
                        raise RuntimeError('NeemHub synchronization did not reach the end of the task')
                    self.meta.update(status='completed',message='Bowl on the table; mug inside the closed drawer. One task completed.',finished_at=time.time())
                    self.persist();return
            raise TimeoutError('Household task exceeded its time limit')
        except BaseException as error:
            for name in ('task','giskard','streaming-client','sim'):
                process=self.processes.pop(name,None)
                if process:self.stop_process(process)
            self.meta.update(status='stopped' if isinstance(error,InterruptedError) else 'failed',message=str(error),finished_at=time.time());self.persist()

    def status(self):
        result=super().status()
        stage=(result.get('state',{}).get('snapshot',{}) or {}).get('sequence_stage')
        result['stage']=EXTRA.get(stage,result['stage']);result['stages']=STAGES
        result.update(viewer_port=8722,live_port=8767,task='household',display_open=bool(self.processes.get('sim')))
        if self.run:
            result['sequence']=legacy.read_json(self.run/'sim/sequence.json',{})
            if result['sequence'].get('status')=='completed': result['stage']=12
            result['files'] += [str(p.relative_to(self.run)) for p in (self.run/'sim/sequence.json',self.run/'task/live-sync.jsonl') if p.exists()]
        return result

    def stop(self):
        if self.worker and self.worker.is_alive():return super().stop()
        if not self.processes:raise RuntimeError('No household display is running')
        for name in ('task','giskard','streaming-client','sim'):
            process=self.processes.pop(name,None)
            if process:self.stop_process(process)


class HouseholdHandler(legacy.Handler):
    def do_GET(self):
        if self.path.split('?')[0] in ('/panel.html','/','/index.html'):
            return self.send_file(Path(__file__).with_name('panel.html'))
        return super().do_GET()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--autostart',action='store_true');a=p.parse_args()
    if not legacy.port_free(8782):raise RuntimeError('Household control service is already running')
    controller=HouseholdController();atexit.register(controller.close)
    server=ThreadingHTTPServer(('127.0.0.1',8782),functools.partial(HouseholdHandler,controller=controller))
    signal.signal(signal.SIGTERM,lambda *args:sys.exit(0))
    if a.autostart:controller.start()
    print('GARMI_HOUSEHOLD_WEB_READY',flush=True)
    server.serve_forever()
if __name__=='__main__':main()
