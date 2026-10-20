"""Make a local video review and measured event timelines; never changes outcomes."""
import html
import json
import math
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/cup4-preppo'
REVIEW=RUN/'failure-review'
LABELS={'success':'成功','attempt_failed':'未抓住','collision':'碰撞','dropped':'掉落'}


def metrics(folder):
    s=json.loads((folder/'summary.json').read_text())
    t=json.loads((folder/'tracking.json').read_text())
    result=s['results'][0]
    initial=result['case']['object_state']
    def event(predicate):
        r=next((r for r in t if predicate(r)),None)
        if r is None:return None
        return dict(seconds=round((r['step']+1)*s['control_dt'],3),
            distance_cm=r['distance']*100,stage=r['plan_stage'],phase=r['phase_after'],
            finger_force=r['finger_force'])
    approaching=[r for r in t if r['phase_after']==0]
    last=approaching[-1]
    q=last['object_task_state'][3:7]
    angle=math.degrees(2*math.acos(min(1,abs(sum(a*b for a,b in zip(q,initial[3:7]))))))
    return dict(case=s['source_case_indices'][0],reference=result['case']['reference'],
        reason=result['physics']['reason'],physics=result['physics'],
        planned_duration=result['planning']['retiming']['seconds'],
        first_finger_contact=event(lambda r:max(r['finger_force'])>.1),
        first_both_fingers=event(lambda r:r['contact']),
        first_cup_shift_5mm=event(lambda r:math.dist(r['object_task_state'][:2],initial[:2])>.005),
        first_close=event(lambda r:r['phase_after']==2),
        max_approach_shift_cm=100*max(math.dist(r['object_task_state'][:2],initial[:2]) for r in approaching),
        approach_final_rotation_deg=angle,approach_final_distance_cm=last['distance']*100,
        approach_final_base_error_mm=last['base_error']*1000)


def main():
    baseline=json.loads((RUN/'tracking-retimed/summary.json').read_text())['results']
    sweep={r['case']:r for r in json.loads((RUN/'gripper-sweep.json').read_text())['results']}
    rows=[]
    cards=[]
    for index in (18,27,41,51,59,70,21,74):
        folder=REVIEW/f'c{index:04d}'
        row=metrics(folder)
        row['original_reason']=baseline[index]['physics']['reason']
        row['nominal_finger_collisions']=sweep[index]['collisions']
        rows.append(row)
        fast=folder/'fast.mp4'
        if not fast.exists():
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(folder/'replay.mp4'),
                '-vf','setpts=PTS/4,fps=30','-an','-c:v','libx264','-crf','23','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(fast)],check=True)
        duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration',
            '-of','default=noprint_wrappers=1:nokey=1',str(fast)],text=True))
        assert duration<10.1,duration
        row['fast_video_seconds']=duration
        def event_text(key):
            e=row[key]
            return '未发生' if e is None else f"{e['seconds']:.2f} 秒"
        nominal=row['nominal_finger_collisions']
        nominal_text='没有检测到手指穿入杯体' if not nominal else (
            '预抓取阶段就检测到手指穿入杯体' if nominal[0]['stage']=='pregrasp'
            else '最后接近阶段检测到手指与杯体相交')
        group='成功对照' if index in (21,74) else '原失败案例'
        caption=f"原批次：{LABELS[row['original_reason']]} → 单独复测：{LABELS[row['reason']]}"
        cards.append(f'''<article id="case-{index}"><h2>c{index:04d} · {group}</h2>
<p class="badge">{html.escape(caption)} · {html.escape(row['reference'])}</p>
<video controls preload="metadata" poster="c{index:04d}/initial-frame.png"><source src="c{index:04d}/fast.mp4" type="video/mp4"></video>
<p><a href="c{index:04d}/replay.mp4">原速录像</a> · <a href="c{index:04d}/summary.json">结果</a> · 画面为 4 倍速</p>
<table><tr><th>首次手指接触杯子</th><td>{event_text('first_finger_contact')}</td></tr>
<tr><th>杯子首次平移超过 5 mm</th><td>{event_text('first_cup_shift_5mm')}</td></tr>
<tr><th>开始闭爪</th><td>{event_text('first_close')}</td></tr>
<tr><th>接近阶段最大平移</th><td>{row['max_approach_shift_cm']:.1f} cm</td></tr>
<tr><th>接近末尾杯子姿态变化</th><td>{row['approach_final_rotation_deg']:.1f}°</td></tr>
<tr><th>接近末尾抓取目标误差</th><td>{row['approach_final_distance_cm']:.1f} cm</td></tr>
<tr><th>重标定后的路径时长</th><td>{row['planned_duration']:.1f} 秒；接近限时仍为 30 秒</td></tr></table>
<p class="note">独立几何检查：{nominal_text}。表内时间均为仿真原速时间。</p></article>''')
    (REVIEW/'events.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GARMI 杯子抓取：失败复查</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#e7edf5;background:#101824;margin:0;padding:30px}
main{max-width:1400px;margin:auto}h1{margin:0}h2{font-size:20px;margin:0 0 8px}.intro{max-width:1000px;color:#b8c7d9}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:22px}article{background:#1b283a;border:1px solid #3b506b;border-radius:12px;padding:20px}
video{width:100%;border-radius:8px;background:#080e17}a{color:#7dccff}.badge{color:#a4d4f7;margin:4px 0 12px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #35475d;padding:7px 0}th{font-weight:500;color:#b4c3d5;width:55%}
.note{font-size:14px;color:#b4c3d5}@media(max-width:500px){body{padding:14px}.grid{grid-template-columns:1fr}article{padding:12px}}</style>
<main><h1>GARMI 杯子抓取：6 个失败案例＋2 个成功对照</h1>
<p class="intro">固定原场景与抓取参考，复用已经修正的底盘接口、轨迹反馈和时间重标定。录像为单环境重新执行，未强行复现某种结局；结果与原 80 环境批次不同的地方明确列出。此页不是 RL 策略录像。</p>
<p class="intro">每段快进录像不超过 10 秒，也可打开原速录像。杯子位移 5 mm 只用于诊断事件标记，没有更改成功失败判据。几何相交不等于一定失败，需结合接触记录和实际结果判断。</p>
<p><a href="report.md">检查结论</a> · <a href="events.json">事件数据</a></p><div class="grid">'''+''.join(cards)+'''</div></main></html>'''
    (REVIEW/'index.html').write_text(page)
    print('REVIEW_READY',[(r['case'],r['reason']) for r in rows])


if __name__=='__main__':main()
