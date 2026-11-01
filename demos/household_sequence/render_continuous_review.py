"""Re-render one recorded household execution with a continuous camera and labels.

This is a presentation replay of measured states, not another physics evaluation.
No policy, planner, or physical success measurement is rerun or altered.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import sys


def smoothstep(value):
    value = min(1., max(0., value))
    return value ** 3 * (10. - 15. * value + 6. * value ** 2)


def camera_at(t, plan):
    """Hold at the counter; move continuously to/from the table during travel."""
    counter, table = plan['counter'], plan['table']
    a, b = plan['outbound_seconds']
    c, d = plan['return_seconds']
    blend = smoothstep((t - a) / (b - a))
    if t >= c:
        blend = 1. - smoothstep((t - c) / (d - c))
    eye, target = tuple([
        (1. - blend) * x + blend * y
        for x, y in zip(counter[key], table[key])
    ] for key in ('eye', 'target'))
    # Bow the travel path into the room to clear the wall corner. The quintic
    # timing keeps position/velocity/acceleration continuous at both holds.
    for i, offset in enumerate(plan.get('travel_eye_offset', [0., 0., 0.])):
        eye[i] += 4. * blend * (1. - blend) * offset
    return eye, target


def label_at(row, bowl_handover):
    s = row['snapshot']
    stage, t = row['stage'], s['simulation_seconds']
    if stage == 'bowl_counter_to_table':
        if s['reference'] and t < bowl_handover:
            return 'RL | Bowl grasp', 'Approach + fixed gripper close and lift', 'rl'
        if s['reference']:
            return 'Giskard | Deliver bowl to dining table', 'Transport, place and retract arm', 'giskard'
        return 'Giskard | Approach kitchen counter', 'Prepare arm and navigate to grasp region', 'giskard'
    if stage == 'return_to_counter_mug':
        return 'Traditional control | Return to counter', 'Base waypoint controller', 'classical'
    if stage == 'mug_rl_approach':
        return 'RL | Mug grasp', 'Approach + fixed gripper close and lift', 'rl'
    captions = {
        'carry_mug_to_drawer': ('Move mug to drawer', 'Base waypoint controller'),
        'open_drawer_left': ('Open drawer with left hand', 'Cartesian IK and gripper control'),
        'place_mug_inside': ('Place mug in drawer', 'Cartesian IK and gripper control'),
        'close_drawer_left': ('Close drawer with left hand', 'Cartesian IK and gripper control'),
    }
    title, detail = captions.get(stage, ('Task complete', 'Bowl on table; mug inside closed drawer'))
    return 'Traditional control | ' + title, detail, 'classical'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--camera-plan', type=Path, required=True)
    parser.add_argument('--preview-times', type=float, nargs='*')
    parser.add_argument('--lead-in-seconds', type=float, default=.4,
                        help='Recorded time to retain before the first planned motion')
    parser.add_argument('--two-groups', action='store_true',
                        help='Presentation grouping: RL or Giskard (all classical control)')
    parser.add_argument('--resolution', type=int, nargs=2, default=[960, 640])
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
    parser.add_argument('--mug-workspace', type=Path, default=Path('/mug'))
    parser.add_argument('--assets', type=Path, default=Path('/demo/assets/mug-drawer'))
    # These must be available before importing Isaac Lab environment modules.
    sys.path[:0] = ['/demo', '/workspace/integration', '/workspace/rl',
                    '/workspace/cram-vrb-lab', '/source/demos/mug_drawer']
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (args.run / 'sim/states.jsonl').read_text().splitlines()]
    sequence = json.loads((args.run / 'sim/sequence.json').read_text())
    report = json.loads((args.run / 'report.json').read_text())
    if (report['status'] != 'completed' or sequence['status'] != 'completed'
            or sequence['scene_initializations'] != 1
            or not any(e['event'] == 'final_verification' and e.get('passed') for e in sequence['events'])):
        raise ValueError('Only a verified, completed single execution can be presented')
    plan = json.loads(args.camera_plan.read_text())
    # Use the actual logged handover, not a new alignment/close threshold.
    events = json.loads((args.run / 'bowl/rl-transport-report.json').read_text())
    wall = [r['wall_time'] for r in rows]
    handover_wall = next(e['time'] for e in events if e['event'] == 'cram_control_returned')
    handover = rows[min(bisect.bisect_left(wall, handover_wall), len(rows)-1)]['snapshot']['simulation_seconds']
    motion_wall = next(e['time'] for e in events if e['event'] == 'motion_start')
    motion_time = rows[min(bisect.bisect_left(wall, motion_wall), len(rows)-1)]['snapshot']['simulation_seconds']
    first_index = next(i for i, row in enumerate(rows)
                       if row['snapshot']['simulation_seconds'] >= motion_time - args.lead_in_seconds)
    app = AppLauncher(args).app
    backend = writer = None
    try:
        import numpy as np
        import imageio.v2 as imageio
        import torch
        from PIL import Image, ImageDraw, ImageFont
        from garmi_cram_rl.contract import Bundle
        from backend import HouseholdBackend
        backend = HouseholdBackend(Bundle(args.workspace / 'results/cram-rl-integration/policy-7500',
                                          args.workspace), args.assets, args.mug_workspace,
                                   args.out, render=True)
        env = backend.env
        env.cfg.viewer.resolution = tuple(args.resolution)
        scale = args.resolution[0] / 960.
        def px(value):
            return round(value * scale)
        font_root = Path('/usr/share/fonts/truetype/dejavu')
        font = ImageFont.truetype(str(font_root / 'DejaVuSans-Bold.ttf'), px(21))
        small = ImageFont.truetype(str(font_root / 'DejaVuSans.ttf'), px(16))
        tiny = ImageFont.truetype(str(font_root / 'DejaVuSans.ttf'), px(13))

        def display_label(row):
            title, detail, kind = label_at(row, handover)
            if kind == 'rl':
                title += ': base + right arm'
            if args.two_groups and kind == 'classical':
                title = title.replace('Traditional control | ', 'Giskard | ', 1)
                kind = 'giskard'
            return title, detail, kind

        def pose(asset, position, quaternion):
            state = asset.data.root_state_w.clone()
            state[0, :3] = state.new_tensor(backend.frame.position_to_task(position))
            state[0, 3:7] = state.new_tensor(backend.frame.quaternion_to_task(quaternion))
            asset.write_root_pose_to_sim(state[:, :7])
            asset.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))

        def restore(row):
            s = row['snapshot']
            for asset, prefix in [(backend.robot, 'base'), (backend.object, 'object'), (backend.mug, 'mug')]:
                pose(asset, s[prefix + '_map_position'], s[prefix + '_map_quaternion_wxyz'])
            joints = backend.robot.data.joint_pos.clone()
            for i, name in enumerate(backend.robot.joint_names):
                joints[0, i] = s['joints'][name]
            backend.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
            drawer = backend.drawer.data.root_state_w.clone()
            drawer[0, :3] = drawer.new_tensor([1.053045 - s['drawer_opening_m'], .39, .8])
            drawer[0, 3:7] = drawer.new_tensor([1., 0., 0., 0.])
            backend.drawer.write_root_pose_to_sim(drawer[:, :7])
            backend.drawer.write_root_velocity_to_sim(torch.zeros_like(drawer[:, 7:]))
            env.sim.forward()
            env.scene.update(0.)
            backend.camera(*camera_at(s['simulation_seconds'], plan))

        def caption(frame, row):
            title, detail, kind = display_label(row)
            im = Image.fromarray(frame).convert('RGBA')
            layer = Image.new('RGBA', im.size)
            draw = ImageDraw.Draw(layer)
            y = im.height - px(87)
            draw.rectangle((0, y, im.width, im.height), fill=(12, 20, 30, 232))
            colors = {'rl': (55, 211, 163, 255), 'giskard': (91, 170, 255, 255), 'classical': (248, 191, 86, 255)}
            draw.rectangle((0, y, px(6), im.height), fill=colors[kind])
            draw.text((px(20), y+px(11)), title, font=font, fill=colors[kind])
            draw.text((px(20), y+px(43)), detail, font=small, fill='white')
            draw.text((im.width-px(225), y+px(64)), 'One recorded execution | 4x', font=tiny, fill=(185, 197, 210))
            return np.asarray(Image.alpha_composite(im, layer).convert('RGB'))

        preview = args.preview_times is not None
        indices = list(range(first_index, len(rows))) if not preview else [
            min(range(len(rows)), key=lambda i: abs(rows[i]['snapshot']['simulation_seconds'] - t))
            for t in args.preview_times]
        # Create the render product before warming up the first recorded view.
        # Otherwise Isaac Lab's first rgb_array call can return a black frame.
        env.render()
        if not preview:
            writer = imageio.get_writer(args.out / 'household-smooth-4x.mp4', fps=40,
                                       macro_block_size=16, codec='libx264', quality=8,
                                       ffmpeg_params=['-movflags', '+faststart'])
        frames = []
        previous = None
        for n, i in enumerate(indices):
            row = rows[i]
            restore(row)
            for _ in range(12 if preview or n == 0 else 2):
                env.sim.render()
            frame = caption(env.render(), row)
            if preview or n % 100 == 0 or n == len(indices)-1:
                imageio.imwrite(args.out / f"frame-{i:05d}-{row['snapshot']['simulation_seconds']:06.1f}.jpg", frame)
            if writer:
                writer.append_data(frame)
            title, detail, kind = display_label(row)
            if title != previous:
                frames.append({'frame': n, 'video_seconds': n / 40., 'simulation_seconds': row['snapshot']['simulation_seconds'],
                               'title': title, 'detail': detail, 'controller': kind,
                               'actual_controller': label_at(row, handover)[2]})
                previous = title
            if n % 100 == 0 or preview:
                print('RENDER', n, '/', len(indices), 'simulation_seconds', row['snapshot']['simulation_seconds'], flush=True)
        if writer:
            writer.close()
            writer = None
        manifest = {'source_run': args.run.name, 'source_state_sha256': hashlib.sha256((args.run/'sim/states.jsonl').read_bytes()).hexdigest(),
                    'presentation': 'Measured-state rerender of one verified continuous execution; no new physics evaluation',
                    'preview': preview, 'frames': len(indices), 'video_fps': 40, 'playback_speed': 4,
                    'resolution': args.resolution,
                    'source_start_frame': indices[0],
                    'source_start_seconds': rows[indices[0]]['snapshot']['simulation_seconds'],
                    'initial_idle_trimmed': not preview,
                    'label_groups': 'RL / Giskard (presentation groups)' if args.two_groups else 'Actual controller categories',
                    'camera_plan': plan, 'controller_segments': frames,
                    'source_completed': True, 'source_scene_initializations': 1}
        (args.out/'render-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if not preview:
            page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>GARMI: one continuous household task</title>
<style>body{font:17px system-ui;background:#101822;color:#e9eff7;max-width:1120px;margin:32px auto;padding:0 20px}video{width:100%;border-radius:12px}p{line-height:1.6}a{color:#8acaff}.rl{color:#37d3a3}.giskard{color:#5baaff}.traditional{color:#f8bf56}</style>
<h1>Counter → dining table → drawer</h1>
<p>Both objects start on the kitchen counter. GARMI delivers the bowl, returns for the mug, opens the drawer, places the mug inside and closes it.</p>
<video controls preload="metadata" src="household-smooth-4x.mp4"></video>
<p>4× playback · Initial waiting removed · Smooth camera movement</p>
<p><span class="rl">RL</span>: approach, fixed gripper closing and lifting. <span class="giskard">Giskard</span>: bowl navigation, transport and placement. <span class="traditional">Traditional control</span>: return waypoints and drawer Cartesian IK.</p>
<p>The camera is re-rendered from the measured states of the successful run; this is not a new policy evaluation or a montage of separate trials.</p>
<p><a href="household-smooth-4x.mp4">Download video</a> · <a href="render-manifest.json">Camera and controller timeline</a></p></html>'''
            if args.two_groups:
                page = page.replace(
                    '<p><span class="rl">RL</span>: approach, fixed gripper closing and lifting. <span class="giskard">Giskard</span>: bowl navigation, transport and placement. <span class="traditional">Traditional control</span>: return waypoints and drawer Cartesian IK.</p>',
                    '<p><span class="rl">RL</span>: base and right-arm approach, fixed gripper closing and lifting. <span class="giskard">Giskard</span>: the classical-control presentation group.</p>'
                    '<p>These are the two requested video labels. Actual execution is unchanged: bowl navigation and placement use Giskard; the return uses waypoint control and the drawer uses Cartesian IK.</p>')
            (args.out/'index.html').write_text(page)
    finally:
        if writer:
            writer.close()
        if backend:
            backend.close()
        app.close()


if __name__ == '__main__':
    main()
