"""Run one continuous household task; owns only the processes it starts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from protocol import verify_bowl_delivery

HERE=Path(__file__).resolve().parent
NATIVE='/home/jovyan/cram-vrb-lab/cognitive_robot_abstract_machine/.venv/bin/python'
SOURCE='source /opt/ros/jazzy/setup.bash; source /home/jovyan/cram-vrb-lab/ros2_ws/install/setup.bash; exec "$@"'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preview-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,ROS_DOMAIN_ID='49',ROS_AUTOMATIC_DISCOVERY_RANGE='LOCALHOST',RMW_IMPLEMENTATION='rmw_fastrtps_cpp',
             PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             PYTHONPATH='/workspace/integration:/workspace/rl:/workspace/cram-vrb-lab:/workspace/scripts:/home/jovyan/cram-vrb-lab:/workspace/third_party/IsaacLab/source/isaaclab:/workspace/third_party/IsaacLab/source/isaaclab_rl',
             FASTRTPS_DEFAULT_PROFILES_FILE='/workspace/integration/fastdds.xml')
    processes={}
    def spawn(name,cmd,extra=None):
        with (args.out/f'{name}.log').open('w') as log:
            proc=subprocess.Popen(cmd,env={**env,**(extra or {})},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[name]=proc;return proc
    def native(script,*extra):return ['bash','-c',SOURCE,'household',NATIVE,str(script),*map(str,extra)]
    def stop(signum,frame):raise InterruptedError('Household run interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    report=dict(status='running',task='bowl-counter-table_then_mug-counter-drawer',episodes=1)
    try:
        sim=spawn('sim',['/isaac-sim/python.sh',str(HERE/'sim.py'),'--out',str(args.out/'sim'),'--headless',
                         *(['--preview-only'] if args.preview_only else [])],
                  {'LD_LIBRARY_PATH':'/usr/local/nvidia/lib64:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib'})
        deadline=time.monotonic()+240
        while 'CRAM_RL_READY' not in (args.out/'sim.log').read_text():
            if sim.poll() is not None:raise RuntimeError('Simulation startup failed; see sim.log')
            if time.monotonic()>deadline:raise TimeoutError('Simulation startup')
            time.sleep(.3)
        if args.preview_only:
            if sim.wait(timeout=60)!=0:raise RuntimeError('Preview failed')
            report['status']='preview_completed';return
        planner=spawn('giskard',native('/workspace/integration/giskard_server.py','--robot','garmi','--scene','garmi_apartment'))
        task=spawn('bowl',native(HERE/'bowl_stage.py',
                                  '--output-dir',args.out/'bowl'))
        deadline=time.monotonic()+1800
        while task.poll() is None:
            if sim.poll() is not None or planner.poll() is not None:raise RuntimeError('Simulation or planner exited during bowl delivery')
            if time.monotonic()>deadline:raise TimeoutError('Bowl delivery')
            time.sleep(.5)
        if task.returncode:raise RuntimeError('Bowl delivery failed; see bowl.log')
        events=json.loads((args.out/'bowl/rl-transport-report.json').read_text())
        verify_bowl_delivery(events)
        (args.out/'continue-mug.json').write_text(json.dumps({'bowl_delivery_verified':True}))
        if sim.wait(timeout=900)!=0:raise RuntimeError('Mug/drawer segment failed; see sim/sequence.json')
        final=json.loads((args.out/'sim/sequence.json').read_text())
        if final['status']!='completed':raise RuntimeError('Missing final verification')
        report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        for proc in reversed(list(processes.values())):
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:continue
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
