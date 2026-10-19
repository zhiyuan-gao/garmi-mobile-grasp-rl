"""Build a review page and trace-aligned clips for every recorded episode."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess


def command(*args):
    subprocess.run(list(map(str, args)), check=True, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes, cards = [], []
    for folder in sorted(args.runs.iterdir()):
        if not (folder / 'summary.json').is_file():
            continue
        summary = json.loads((folder / 'summary.json').read_text())
        case = summary['case']
        if summary['status'] == 'running':
            continue
        source = folder / 'sim/replay.mp4'
        if not source.is_file():
            continue
        trace = [json.loads(line) for line in (folder / 'sim/states.jsonl').read_text().splitlines()]
        meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=nb_frames,r_frame_rate,duration', '-of', 'json', str(source)]))['streams'][0]
        assert len(trace) == int(meta['nb_frames']), (folder, len(trace), meta)
        assert meta['r_frame_rate'] == '10/1'
        active = [i for i, row in enumerate(trace) if row['owner'] == 'rl']
        start = active[0] if active else max(0, len(trace) - 100)
        end = next((i for i in range(start + 1, len(trace)) if trace[i]['owner'] != 'rl'), len(trace)) if active else len(trace)
        fixed = next((i for i in range(start, end) if trace[i]['snapshot']['phase'] > 0), end)
        prefix = case['id']
        full, grasp, quick, poster = [args.out / f'{prefix}-{name}' for name in ('full.mp4', 'grasp.mp4', 'quick.mp4', 'poster.jpg')]
        shutil.copy2(source, full)
        command('ffmpeg', '-y', '-loglevel', 'error', '-i', source, '-an', '-vf',
                f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', grasp)
        speed = len(trace) / 10 / 9.5
        command('ffmpeg', '-y', '-loglevel', 'error', '-i', source, '-an', '-vf',
                f'setpts=PTS/{speed:.9f},fps=30', '-t', '9.6', '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', quick)
        command('ffmpeg', '-y', '-loglevel', 'error', '-ss', str(max(0, (fixed-start)/10 - .2)),
                '-i', grasp, '-frames:v', '1', poster)
        item = dict(summary, source_run=str(folder), fps=10, source_frames=len(trace),
                    clip_start_frame=start, clip_end_frame=end, grasp_seconds=(end-start)/10,
                    rl_started=bool(active),
                    rl_approach_seconds=(fixed-start)/10 if active else 0.,
                    fixed_finish_seconds=(end-fixed)/10 if active else 0.,
                    measured_rl_start=trace[start]['snapshot']['base_map_position'] if active else None,
                    measured_bowl=trace[start]['snapshot']['object_map_position'],
                    video_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    full_video=full.name, grasp_video=grasp.name, quick_video=quick.name, poster=poster.name)
        episodes.append(item)
        passed = summary['status'] == 'completed'
        placement = summary.get('placement')
        error = f"Placement error: {placement['horizontal_error_m']*1000:.1f} mm." if placement else html.escape(summary.get('error') or 'No placement result.')
        coords = lambda value: ', '.join(f'{x:.2f}' for x in value[:2]) if value else 'not started'
        status = 'Completed' if passed else ('Grasp succeeded; later stage failed' if summary.get('grasp_succeeded') else 'Failed')
        if not active and summary.get('navigation_arrival'):
            status = 'RL not started; handover rejected'
        clip_label = 'Approach, close and lift' if active else 'Task excerpt; RL was not started'
        arrival = summary.get('navigation_arrival') or {}
        retry = f"<p>Follow-up to {html.escape(case['retry_of'])}: {html.escape(case['reason'])}</p>" if case.get('retry_of') else ''
        cards.append(f'''<article><h2>{html.escape(prefix)} <span class="{'ok' if passed else 'failed'}">{status}</span></h2>
<video controls playsinline preload="metadata" poster="{poster.name}"><source src="{grasp.name}" type="video/mp4"></video>
<p class="caption">{clip_label} · {item['grasp_seconds']:.1f} s · original simulation playback rate.<br>RL approach: {item['rl_approach_seconds']:.1f} s; fixed finishing controller: {item['fixed_finish_seconds']:.1f} s.</p>
<dl><dt>Bowl initial XY</dt><dd>({coords(case['bowl'])}) m</dd><dt>Navigation start XY</dt><dd>({coords(case['navigation_start'])}) m</dd><dt>RL start XY, requested</dt><dd>({coords(case['rl_start'])}) m</dd><dt>Navigation arrival XY</dt><dd>({coords(arrival.get('actual'))}) m</dd><dt>RL start XY, measured</dt><dd>{'(' + coords(item['measured_rl_start']) + ') m' if active else 'Not started'}</dd></dl>
<p>{error}</p>{retry}<p><a href="{full.name}">Full task recording</a> · <a href="{quick.name}">Under-10-second full-task preview</a></p>
<details><summary>Play full task</summary><video controls playsinline preload="metadata" src="{full.name}"></video></details></article>''')
    successes = sum(e['status'] == 'completed' for e in episodes)
    (args.out / 'episodes.json').write_text(json.dumps(episodes, indent=2) + '\n')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GARMI · Left-counter grasp episodes</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;color:#edf4fc;background:#07111f}*{box-sizing:border-box}body{margin:0}main{max-width:1280px;margin:auto;padding:30px 24px 60px}a{color:#67d9c7}h1{font-size:clamp(28px,4vw,44px)}p{line-height:1.6}.intro{max-width:950px;color:#bac8d9}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{padding:18px;background:#111d30;border:1px solid #293b58;border-radius:12px}h2{font-size:19px;margin:0 0 16px}h2 span{display:block;font-size:13px;margin-top:8px}.ok{color:#67d9c7}.failed{color:#ffbe82}video{width:100%;display:block;background:#02070e;border-radius:8px}.caption{font-size:13px;color:#bac8d9}dl{display:grid;grid-template-columns:1.25fr 1fr;gap:6px;font-size:13px}dt{color:#bac8d9}dd{margin:0}summary{cursor:pointer;margin:15px 0}@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:24px 14px}dl{grid-template-columns:1fr 1fr}}</style></head><body><main>
<p><a href="../">GARMI demo</a></p><h1>A clearer view of mobile grasping</h1>
<p class="intro">The left-counter setups vary both the bowl position and the robot's initial positions. The camera views the bowl from the opposite side of the wrist. Every case uses the same trained checkpoint 7500 and reference bowl-r020. CRAM navigates to the specified handover position; RL then controls the mobile base and right arm. Closing and lifting use the fixed controller, followed by CRAM transport and placement. Case 05 is an explicitly marked follow-up to case 04 with the handover goal moved inward; the original rejected attempt is retained.</p>
'''
    page += f'<p class="intro">Recorded results: {successes}/{len(episodes)} complete tasks. These are demonstration cases, not a success-rate evaluation. Coordinates use the apartment map frame; playback follows simulation time, not wall-clock time. <a href="episodes.json">Run details and clip provenance</a>.</p><div class="grid">'
    page += '\n'.join(cards) + '</div></main></body></html>\n'
    (args.out / 'index.html').write_text(page)
    print(json.dumps(dict(episodes=len(episodes), completed=successes, page=str(args.out / 'index.html'))))


if __name__ == '__main__':
    main()
