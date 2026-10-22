"""Run the isolated CRAM/RL single-scene bridge after training is idle."""
import argparse
import json
from pathlib import Path
import sys


def ensure_training_idle(workspace):
    path = workspace / 'results/rl-trials/trial-17-bowl-kitchen-40refs-expert-ppo/supervisor.json'
    if path.exists() and json.loads(path.read_text()).get('status') == 'running':
        raise RuntimeError('Current training is running. This integration simulator will not start alongside it.')


def main():
    root = Path(__file__).resolve().parents[1]
    ensure_training_idle(root)  # Before importing AppLauncher or touching CUDA.
    sys.path[:0] = [str(root / 'rl'), str(root / 'cram-vrb-lab')]
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base-map', type=float, nargs=3, default=[0., 5.8, 1.5707963267948966])
    parser.add_argument('--bowl-map', type=float, nargs=2, default=[.2, 7.4])
    parser.add_argument('--full-scene', action='store_true')
    parser.add_argument('--drawer-scene', action='store_true', help='Movable left arm and upper kitchen drawer')
    parser.add_argument('--live-preview', action='store_true', help='Publish an atomic JPEG for the separate web demo')
    parser.add_argument('--desktop-stream', action='store_true',
                        help='Use the original GARMI 640x360 WebRTC viewport without video recording')
    parser.add_argument('--allow-episode-reset', action='store_true',
                        help='Allow scene reset between tasks without restarting Isaac Sim')
    parser.add_argument('--probe-grasp-views', action='store_true', help='Save alternate camera images at close/lift, without advancing physics')
    parser.add_argument('--max-seconds', type=float, default=600., help='Maximum wall time after startup')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.allow_episode_reset and (args.enable_cameras or args.drawer_scene):
        parser.error('--allow-episode-reset requires the non-recording grasp simulator')
    if args.desktop_stream and (args.enable_cameras or args.drawer_scene):
        parser.error('--desktop-stream cannot be combined with --enable_cameras or --drawer-scene')
    if args.desktop_stream:
        # Isaac Lab's bundled livestream option uses an extension removed in Sim 6.1.
        # Enable the same supported extension as the original GARMI demo below.
        args.headless = True
        args.livestream = 0
    from garmi_cram_rl.contract import Bundle
    bundle = Bundle(args.bundle, root)
    args.out.mkdir(parents=True, exist_ok=False)
    launch_options = {'renderer': 'RaytracedLighting'} if args.live_preview or args.desktop_stream else {}
    if args.desktop_stream:
        launch_options.update(width=640, height=360, display_options=3286)
    app = AppLauncher(args, **launch_options).app
    from isaacsim.core.utils.extensions import enable_extension
    if args.desktop_stream:
        app.set_setting('/app/livestream/enabled', True)
        app.set_setting('/isaaclab/render/active_viewport', True)
        app.set_setting('/physics/fabricUpdateTransformations', True)
        app.set_setting('/app/window/drawMouse', True)
        app.set_setting('/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize', True)
        enable_extension('omni.kit.livestream.app')
    enable_extension('isaacsim.ros2.bridge')
    import rclpy
    import time
    from garmi_cram_rl.lab_backend import LabBackend
    from garmi_cram_rl.ros_bridge import Bridge
    rclpy.init()
    if args.drawer_scene:
        from garmi_cram_rl.drawer_backend import DrawerBackend
        backend = DrawerBackend(bundle, args.base_map, args.bowl_map, render=args.enable_cameras)
    else:
        def configure(cfg):
            if args.live_preview or args.desktop_stream:
                # The web page owns the run controls; leave the desktop viewport unobstructed.
                cfg.ui_window_class_type = None
            if args.desktop_stream:
                cfg.viewer.resolution = (640, 360)
                # Match the original 25 Hz render cadence, retaining trained physics.
                cfg.sim.render_interval = max(1, round(.04 / cfg.sim.dt))
        backend = LabBackend(bundle, args.base_map, args.bowl_map, render=args.enable_cameras,
                             full_scene=args.full_scene, configure=configure, display=args.desktop_stream)
    if args.desktop_stream:
        from garmi_cram_rl.viewport import configure_streamed_navigation
        configure_streamed_navigation()
    if args.live_preview and not args.headless:
        import omni.ui
        for title in ('IsaacLab', 'Property'):
            window = omni.ui.Workspace.get_window(title)
            if window:
                window.visible = False
    node = Bridge(backend, args.out / 'events.jsonl')
    import signal
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    writer = None
    frame = None
    if args.enable_cameras:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.out / 'replay.mp4', fps=10, macro_block_size=16)
    trace = (args.out / 'states.jsonl').open('w', buffering=1)
    deadline = time.monotonic() + args.max_seconds
    if args.desktop_stream:
        print('Display: WebRTC 640x360, target rendering 25 Hz; video and JPEG capture disabled.', flush=True)
    print('CRAM_RL_READY', flush=True)
    probed_phases = set()
    def reset_episode(output):
        nonlocal trace, deadline
        destination = Path(output).resolve()
        if destination.name != 'sim' or destination.parent.parent != args.out.resolve().parent.parent:
            raise ValueError('Episode output must be a new sibling run directory')
        destination.mkdir(parents=True, exist_ok=False)
        node.reset_episode(destination / 'events.jsonl')
        trace.close()
        trace = (destination / 'states.jsonl').open('w', buffering=1)
        args.out = destination
        deadline = time.monotonic() + args.max_seconds
        probed_phases.clear()
        print(f'CRAM_RL_RESET_READY {destination}', flush=True)
    if args.allow_episode_reset:
        node.reset_handler = reset_episode
    try:
        while app.is_running() and not stopped and time.monotonic() < deadline:
            started = time.monotonic()
            for _ in range(16):
                rclpy.spin_once(node, timeout_sec=0.)
            node.step()
            node.publish()
            if backend.control_step % 2 == 0:
                trace.write(json.dumps(dict(wall_time=time.time(), **node.session.status(),
                                            snapshot=backend.snapshot(), fault=node.fault)) + '\n')
                if writer is not None:
                    frame = backend.env.render()
                    writer.append_data(frame)
                    phase = int(backend.env.fixed_finish.phase[0])
                    if (args.probe_grasp_views and node.session.owner == 'rl'
                            and phase in (1, 3) and phase not in probed_phases):
                        probed_phases.add(phase)
                        eye, target = backend.camera_map
                        bowl = backend.snapshot()['object_map_position']
                        views = {'back_left': [-.55, .28, .85], 'left_side': [-.75, 0., .50],
                                 'overhead': [.05, .03, 1.25], 'back_right': [.55, .28, .70]}
                        for label, offset in views.items():
                            backend.camera([p + d for p, d in zip(bowl, offset)],
                                           [bowl[0], bowl[1], bowl[2] + .12])
                            for _ in range(3):
                                backend.env.sim.render()
                            imageio.imwrite(args.out / f'camera-phase-{phase}-{label}.png', backend.env.render())
                        backend.camera(eye, target)
                        for _ in range(3):
                            backend.env.sim.render()
                    if backend.control_step == 20:
                        imageio.imwrite(args.out / 'initial-frame.png', frame)
                    if args.live_preview and backend.control_step % 10 == 0:
                        preview = args.out / 'preview.next.jpg'
                        imageio.imwrite(preview, frame, quality=80)
                        preview.replace(args.out / 'preview.jpg')
                    if args.drawer_scene and backend.control_step % 100 == 0:
                        imageio.imwrite(args.out / 'latest-frame.png', frame)
            time.sleep(max(0., backend.env.step_dt - (time.monotonic() - started)))
    finally:
        trace.close()
        if writer is not None:
            if frame is not None:
                imageio.imwrite(args.out / 'final-frame.png', frame)
            writer.close()
        node.destroy_node()
        backend.close()
        rclpy.shutdown()
        app.close()


if __name__ == '__main__':
    main()
