"""Counter-to-table demo with NeemHub observation; the ordinary demo stays unchanged."""
import argparse
import math
from pathlib import Path
import threading

from cram_demo import GraspDemo, cancel_motion
from mobile_grasp import execute_single, real_robot, matrix


class WebDemo(GraspDemo):
    viewer = None

    def setup_object(self):
        super().setup_object()
        from cramera.live.visualization import LiveVisualization
        self.viewer = LiveVisualization(self.world, port=8766).start()
        self.record('viewer_ready', port=8766)

    def run_motion(self, label, action, timeout=75):
        # Same motion execution and watchdog as MobileGrasp, with the existing
        # cramera callback attached to the plan before it runs.
        self.record('motion_start', label=label)
        timed_out = threading.Event()
        def stop():
            timed_out.set()
            cancel_motion(self.context)
        timer = threading.Timer(timeout, stop)
        timer.daemon = True
        timer.start()
        try:
            with real_robot(collision_avoidance=True):
                node = execute_single(action, context=self.context)
                plan = node.plan
                plan.node_callbacks.append(self.viewer.plan_callback(plan))
                node.perform()
        finally:
            timer.cancel()
        if timed_out.is_set():
            raise TimeoutError(label)
        self.record('motion_done', label=label, tcp=matrix(self.tool.global_pose).tolist(),
                    base=matrix(self.robot.root.global_pose).tolist(), object=self.object_pose())

    def close(self):
        if self.viewer is not None:
            # Persist this run's recording before shutting down its live bridge.
            from cramera.live.recording_bundle import finalize_recording
            bridge = self.viewer.bridge
            if bridge.recording is not None:
                finalize_recording(bridge, bridge.recording)
            self.viewer.stop()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    options.reference = 'bowl-r020'
    options.base = [.25, 6.35, math.pi / 2]
    options.place_base = [.90, 5.85, -.85]
    options.target = [1.6, 5.32, .80]
    options.stage, options.carry = 'rl-transport', 'physical'
    options.transport = True
    options.resume_place = False
    options.retreat = None
    demo = WebDemo(options, output=options.output_dir)
    try:
        demo.run()
        demo.record('completed', stage=options.stage)
    except BaseException as error:
        if hasattr(demo, 'context'):
            cancel_motion(demo.context)
        demo.output.mkdir(parents=True, exist_ok=True)
        demo.record('failed', error=repr(error))
        raise
    finally:
        if hasattr(demo, 'executor'):
            demo.close()


if __name__ == '__main__':
    main()
