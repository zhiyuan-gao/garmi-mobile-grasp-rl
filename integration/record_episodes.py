"""Record a fixed list of counter-to-table episodes using the existing CRAM/RL task.

Run inside the prepared demo container. No policy, reward, physics or planner
parameters are changed; cases specify only initial poses, grasp reference and camera.
"""
import argparse
import json
import os
from pathlib import Path
import time

from web_server import Controller, ROOT, ensure_training_idle, external_integration_running, tail


def run_case(case, out, probe=False):
    ensure_training_idle(ROOT)
    if external_integration_running():
        raise RuntimeError('Another integration simulation or task is active')
    folder = out / case['id']
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'case.json').write_text(json.dumps(case, indent=2) + '\n')
    controller = Controller(services=False)
    controller.run = folder
    start = time.time()
    summary = dict(case=case, status='running', started_at=start)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    try:
        env = controller.env()
        env.update(FASTRTPS_DEFAULT_PROFILES_FILE=str(ROOT / 'integration/fastdds.xml'),
                   LD_LIBRARY_PATH='/usr/local/nvidia/lib64:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib')
        command = ['/isaac-sim/python.sh', str(ROOT / 'integration/sim.py'),
                   '--bundle', str(ROOT / 'results/cram-rl-integration/policy-7500'),
                   '--out', str(folder / 'sim'), '--full-scene', '--headless', '--enable_cameras',
                   '--live-preview', '--max-seconds', '900',
                   '--base-map', *map(str, case['navigation_start']),
                   '--bowl-map', *map(str, case['bowl'])]
        if probe:
            command.append('--probe-grasp-views')
        controller.spawn('sim', command, env=env)
        controller.wait_ready('sim', lambda: 'CRAM_RL_READY' in tail(folder / 'sim.log'), 240)
        controller.spawn('giskard', controller.native(ROOT / 'integration/giskard_server.py',
                         '--robot', 'garmi', '--scene', 'garmi_apartment'))
        task = controller.spawn('task', controller.native(ROOT / 'integration/cram_demo.py',
                   '--transport', '--reference', case['reference'], '--output-dir', folder / 'task',
                   '--base', *case['rl_start'], '--grasp-camera-offset', *case['camera_offset']))
        deadline = time.monotonic() + 720
        while task.poll() is None:
            time.sleep(.5)
            if time.monotonic() > deadline:
                raise TimeoutError('Episode exceeded 12 minutes')
            if any(controller.processes[n].poll() is not None for n in ('sim', 'giskard')):
                raise RuntimeError('Simulator or Giskard stopped')
        events = json.loads((folder / 'task/rl-transport-report.json').read_text())
        summary['grasp_succeeded'] = any(e['event'] == 'rl_grasp_result' for e in events)
        summary['placement'] = next((e for e in events if e['event'] == 'transport_result'), None)
        summary['navigation_arrival'] = next((e for e in events if e['event'] == 'navigation_arrival'), None)
        passed = (task.returncode == 0 and any(e['event'] == 'completed' for e in events)
                  and summary['placement'] and summary['placement']['passed'])
        summary['status'] = 'completed' if passed else 'failed'
        summary['error'] = next((e.get('error') for e in reversed(events) if e['event'] == 'failed'), None)
    except Exception as error:
        summary.update(status='failed', error=repr(error))
    finally:
        for name in ('task', 'giskard', 'sim'):
            process = controller.processes.pop(name, None)
            if process:
                controller.stop_process(process)
        summary.update(finished_at=time.time(), elapsed_seconds=time.time() - start)
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--probe-grasp-views', action='store_true')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        if args.only and case['id'] not in args.only:
            continue
        print('EPISODE_START', case['id'], flush=True)
        result = run_case(case, args.out, args.probe_grasp_views)
        results.append(result)
        print('EPISODE_RESULT', json.dumps(result), flush=True)
    (args.out / ('batch-' + str(os.getpid()) + '.json')).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
