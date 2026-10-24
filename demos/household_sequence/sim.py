"""Persistent simulation for exactly one bowl -> mug -> drawer task."""
import argparse
from pathlib import Path
import sys
import time
import json
import signal

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),'/workspace/integration','/workspace/rl','/workspace/cram-vrb-lab',str(HERE.parent/'mug_drawer')]
# /demo is also supported by the isolated Docker launcher.
if Path('/source/demos/mug_drawer').exists():sys.path.insert(0,'/source/demos/mug_drawer')
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--workspace',type=Path,default=Path('/workspace'))
p.add_argument('--mug-workspace',type=Path,default=Path('/mug'))
p.add_argument('--assets',type=Path,default=HERE/'assets/mug-drawer')
p.add_argument('--max-seconds',type=float,default=2400.)
p.add_argument('--preview-only',action='store_true')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.enable_cameras=True
a.out.mkdir(parents=True,exist_ok=False)
app=AppLauncher(a).app
from isaacsim.core.utils.extensions import enable_extension
enable_extension('isaacsim.ros2.bridge')
import rclpy
import imageio.v2 as imageio
from garmi_cram_rl.contract import Bundle
from garmi_cram_rl.ros_bridge import Bridge
from backend import HouseholdBackend
from protocol import verify_bowl_delivery

stopped=False

def stop(signum,frame):
    global stopped
    stopped=True

signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
backend=node=writer=None
try:
    bundle=Bundle(a.workspace/'results/cram-rl-integration/policy-7500',a.workspace)
    backend=HouseholdBackend(bundle,a.assets,a.mug_workspace,a.out,render=True)
    # Reject any accidental later attempt to reset the shared scene.
    def no_reset(*args,**kwargs):raise RuntimeError('A continuous household task cannot reset the scene')
    backend.env.reset=no_reset
    backend.env._reset_idx=no_reset
    rclpy.init()
    node=Bridge(backend,a.out/'events.jsonl')
    backend.camera([-1.2,4.4,2.7],[.7,6.3,1.])
    backend.env.render()
    for _ in range(20):backend.env.sim.render()
    imageio.imwrite(a.out/'initial.png',backend.env.render())
    print('CRAM_RL_READY',flush=True)
    if not a.preview_only:
        writer=imageio.get_writer(a.out/'replay.mp4',fps=10,macro_block_size=16)
        sequence=None;deadline=time.monotonic()+a.max_seconds
        with (a.out/'states.jsonl').open('w',buffering=1) as trace:
            while app.is_running() and not stopped and time.monotonic()<deadline:
                start=time.monotonic()
                for _ in range(16):rclpy.spin_once(node,timeout_sec=0.)
                if sequence is None:
                    trigger=a.out.parent/'continue-mug.json'
                    if trigger.exists():
                        report=json.loads((a.out.parent/'bowl/rl-transport-report.json').read_text())
                        verify_bowl_delivery(report)
                        # Old queued ROS commands cannot compete with the second segment.
                        node.classical=lambda:False
                        sequence=backend.continue_with_mug()
                if sequence is None:node.step()
                else:
                    try:next(sequence)
                    except StopIteration:break
                node.publish()
                if backend.control_step%2==0:
                    trace.write(json.dumps(dict(wall_time=time.time(),snapshot=backend.snapshot(),stage=backend.stage,fault=node.fault))+'\n')
                    frame=backend.env.render();writer.append_data(frame)
                    if backend.control_step%10==0:
                        imageio.imwrite(a.out/'preview.next.jpg',frame)
                        (a.out/'preview.next.jpg').replace(a.out/'preview.jpg')
                if node.fault:raise RuntimeError(node.fault)
                time.sleep(max(0.,backend.env.step_dt-(time.monotonic()-start)))
        if backend.sequence_report['status']!='completed':
            raise RuntimeError('Stopped before completing the continuous task')
    imageio.imwrite(a.out/'final.png',backend.env.render())
except BaseException as error:
    if backend:
        backend.sequence_report.update(status='failed',error=repr(error))
        backend.record('failed',error=repr(error))
        imageio.imwrite(a.out/'failed.png',backend.env.render())
    raise
finally:
    if writer:writer.close()
    if node:node.destroy_node()
    if backend:backend.close()
    if rclpy.ok():rclpy.shutdown()
    app.close()
