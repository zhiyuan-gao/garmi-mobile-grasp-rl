"""Export a verified continuous run as separate normal-speed and 4x MP4 files."""
import argparse, hashlib, html, json, shutil, subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--ffmpeg',default=shutil.which('ffmpeg') or 'ffmpeg');p.add_argument('--mp4-only',action='store_true',help='Write only the 4x MP4, without HTML or normal-speed copies');args=p.parse_args()
r=json.loads((args.run/'report.json').read_text())
if r['status']!='completed' or r.get('debug_resumed'):
 raise ValueError('Export requires a completed continuous run starting with RL inference')
events=r['events'];grasp=next(e for e in events if e['event']=='rl_grasp');opened=next(e for e in events if e['event']=='drawer_open')
assert grasp['status']=='success' and opened['opening']>.28
assert r['mug_inside'] and abs(r['drawer_opening_m'])<.02 and r['mug_speed']<.02
args.out.mkdir(parents=True,exist_ok=False)
if args.mp4_only:
 target=args.out/'mug-to-drawer-full-scene-4x.mp4'
 subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(args.run/'raw.mp4'),'-vf',"setpts=PTS/4,scale=1600:900,fps=30",'-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-an','-movflags','+faststart',str(target)],check=True)
 print(target);raise SystemExit(0)
for name,filters in [('mug-to-drawer-1x.mp4',['-c','copy']),('mug-to-drawer-4x.mp4',['-vf','setpts=PTS/4,fps=30','-c:v','libx264','-crf','20','-preset','medium','-pix_fmt','yuv420p'])]:
 subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(args.run/'raw.mp4'),*filters,'-an','-movflags','+faststart',str(args.out/name)],check=True)
poster=next(iter(sorted(args.run.glob('*-right_cleared.png'))),args.run/'final.png')
shutil.copy2(poster,args.out/'poster.png')
manifest=dict(checkpoint_sha256=r['checkpoint_sha256'],controller=r['controller'],continuous=True,simulation_seconds=events[-1]['sim_seconds'],drawer_opening_m=r['drawer_opening_m'],mug_inside=r['mug_inside'],mug_speed_m_s=r['mug_speed'],videos={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.out.glob('*.mp4')})
(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(args.out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GARMI — Mug to Drawer</title><style>body{font:18px system-ui;background:#101923;color:#eaf0f5;max-width:1080px;margin:40px auto;padding:0 20px}video{width:100%;border-radius:12px;background:#000}a{color:#86c8ff;margin-right:24px}p{line-height:1.6;color:#b8c8d7}</style><h1>GARMI: Mug to Drawer</h1><p>RL mug body grasp → Left-hand drawer opening → Right-hand placement → Left-hand drawer closing</p><video controls playsinline preload="metadata" poster="poster.png" src="mug-to-drawer-4x.mp4"></video><p>4× playback</p><p><a href="mug-to-drawer-4x.mp4" target="_blank">Open 4× video</a><a href="mug-to-drawer-1x.mp4" target="_blank">Open normal-speed video</a></p><p>The right-hand grasp uses the mug PPO 6500 policy. Subsequent motions use Cartesian IK control. The mug and drawer move through physical contact.</p></html>''')
print(args.out)
