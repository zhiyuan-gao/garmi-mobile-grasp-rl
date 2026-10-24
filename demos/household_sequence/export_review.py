"""Export a completed continuous task as a 4x video and a local review page."""
import argparse
import html
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / 'report.json').read_text())
    sequence = json.loads((run / 'sim/sequence.json').read_text())
    if report['status'] != 'completed' or sequence['status'] != 'completed':
        raise ValueError('A completed continuous run is required')
    if report['episodes'] != 1 or sequence['scene_initializations'] != 1:
        raise ValueError('Expected one task and one scene initialization')
    if not any(e['event'] == 'final_verification' and e.get('passed') for e in sequence['events']):
        raise ValueError('Final verification is missing')
    output = run / 'review'
    output.mkdir(exist_ok=False)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(run / 'sim/replay.mp4'),
                    '-vf', 'setpts=PTS/4,fps=30', '-an', '-c:v', 'libx264',
                    '-crf', '21', '-movflags', '+faststart',
                    str(output / 'household-4x.mp4')], check=True)
    summary = {
        'status': 'completed', 'episodes': 1,
        'initial_bowl_map': sequence['events'][0]['bowl_map'],
        'initial_mug_map': sequence['events'][0]['mug_map'],
        'simulation_seconds': sequence['events'][-1]['simulation_seconds'],
        'mug_inside': sequence['mug_inside'],
        'drawer_opening_m': sequence['drawer_opening_m'],
        'bowl_shift_after_delivery_m': sequence['bowl_shift'],
        'mug_speed_m_s': sequence['mug_speed'],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    content = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>GARMI continuous household task</title>
<style>body{font:17px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#f6f7fa;color:#17212e}video{width:100%;background:#111;border-radius:10px}pre{white-space:pre-wrap;font-size:14px;padding:18px;background:white;border-radius:10px}p{line-height:1.6}a{color:#285caf}</style>
<h1>GARMI: counter → dining table → drawer</h1>
<p>The bowl and mug both start on the kitchen counter. GARMI delivers the bowl to the dining table, returns for the mug, opens the drawer with its left hand, places the mug inside with its right hand, and closes the drawer. One continuous task, with no scene reset.</p>
<video controls preload="metadata" src="household-4x.mp4"></video>
<p>Playback: 4×. <a href="../sim/replay.mp4">Original speed</a> · <a href="../sim/sequence.json">Stage measurements</a></p>
<h2>Verified result</h2><pre>''' + html.escape(json.dumps(summary, indent=2)) + '</pre></html>\n'
    (output / 'index.html').write_text(content)
    print(output / 'index.html')


if __name__ == '__main__':
    main()
