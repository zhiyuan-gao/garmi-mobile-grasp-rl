"""Counter-to-table demo with NeemHub observation; the ordinary demo stays unchanged."""
import argparse
import math
import os
from pathlib import Path
import shutil

from cram_demo import GraspDemo, cancel_motion
from mobile_grasp import execute_single, real_robot, matrix
from motion_watchdog import motion_deadline


class WebDemo(GraspDemo):
    viewer = None

    def setup_object(self):
        super().setup_object()
        from cramera.live.visualization import LiveVisualization
        from cramera.scene_presentation import ScenePresentation
        port = int(os.environ.get('GARMI_LIVE_PORT', '8766'))
        self.viewer = LiveVisualization(self.world, port=port)
        self.viewer.bridge.presentation = ScenePresentation(
            camera={'position': [1.2, 4.95, 2.5], 'target': [.4, 6.6, 1.1]})
        self.viewer.start()
        self.record('viewer_ready', port=port)

    def run_motion(self, label, action, timeout=75):
        # Same motion execution and watchdog as MobileGrasp, with the existing
        # cramera callback attached to the plan before it runs.
        self.record('motion_start', label=label)
        try:
            seconds = timeout * float(os.environ.get('GARMI_MOTION_TIMEOUT_SCALE', '1'))
            with motion_deadline(seconds, label), real_robot(collision_avoidance=True):
                node = execute_single(action, context=self.context)
                plan = node.plan
                plan.node_callbacks.append(self.viewer.plan_callback(plan))
                node.perform()
        except TimeoutError:
            cancel_motion(self.context)
            raise
        self.record('motion_done', label=label, tcp=matrix(self.tool.global_pose).tolist(),
                    base=matrix(self.robot.root.global_pose).tolist(), object=self.object_pose())

    def close(self):
        if self.viewer is not None:
            # Persist this run's recording before shutting down its live bridge.
            from cramera.live.recording_bundle import finalize_recording
            bridge = self.viewer.bridge
            if bridge.recording is not None:
                name = finalize_recording(bridge, bridge.recording)
                archive = os.environ.get('GARMI_RECORDING_ARCHIVE')
                if name and archive:
                    from cramera.paths import local_scenes_directory
                    source = local_scenes_directory() / name
                    destination = Path(archive) / name
                    if source.resolve() != destination.resolve():
                        shutil.copytree(source, destination, dirs_exist_ok=True)
            self.viewer.stop()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    options = parser.parse_args()
    if options.output_dir.exists():
        raise FileExistsError(options.output_dir)
    # The native navigation goal also has its own wall-clock watchdog.
    from cram_vrb_lab.robots.garmi import motions
    motions.MOVE_TIMEOUT *= float(os.environ.get('GARMI_MOTION_TIMEOUT_SCALE', '1'))
    options.reference = 'bowl-r020'
    options.base = [.25, 6.35, math.pi / 2]
    options.place_base = [.90, 5.85, -.85]
    options.target = [1.6, 5.32, .80]
    # Keep the close-up inside the apartment, in front of the counter window.
    options.grasp_camera_offset = [.65, -.6, .75]
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
