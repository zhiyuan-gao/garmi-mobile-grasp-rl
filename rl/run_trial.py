"""Host-side supervisor for a frozen PPO trial; no online parameter tuning."""
import argparse
import collections
import csv
import hashlib
import html
import json
import os
import signal
import subprocess
import shutil
import time
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = ['bash', str(ROOT / 'scripts/container_exec.sh')]
parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--config", default="configs/task.yaml")
parser.add_argument("--title", default="GARMI PPO 试训")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--updates", type=int, default=1000)
parser.add_argument("--eval-interval", type=int, default=100)
parser.add_argument("--eval-updates", type=int, nargs="+")
parser.add_argument("--eval-envs", type=int, default=100)
parser.add_argument("--paired-eval-cases", help="Additional fixed kitchen evaluation set")
parser.add_argument("--paired-eval-envs", type=int, default=120)
parser.add_argument("--paired-eval-seed", type=int, default=20261018)
parser.add_argument("--resume")
parser.add_argument("--allow-env-resize", action="store_true",
                    help="Allow an audited change in environment count when resuming")
parser.add_argument("--defer-evaluation", action="store_true",
                    help="Final-checkpoint-only evaluation after releasing the training GPU process")
parser.add_argument("--init-checkpoint")
parser.add_argument("--wall-hours",type=float,default=0.)
parser.add_argument("--final-eval-envs",type=int,default=512)
parser.add_argument("--final-eval-splits", nargs="+", choices=("train", "heldout", "all"),
                    default=["train", "heldout"])
parser.add_argument("--video-limit",type=int,default=0,help="Zero records one success/failure per object")
parser.add_argument("--skip-videos", action="store_true")
args = parser.parse_args()
if not args.skip_videos:
    parser.error("Use --skip-videos; record this kitchen task with evaluate_kitchen.py")
if args.allow_env_resize and not args.resume:
    parser.error('--allow-env-resize requires --resume')
if args.defer_evaluation and set(args.eval_updates or []) != {args.updates}:
    parser.error('--defer-evaluation requires --eval-updates with only the final update')
if args.eval_updates and any(u < 1 or u > args.updates for u in args.eval_updates):
    parser.error("Evaluation updates must be within the training budget")
out = (ROOT / args.out).resolve()
relative = out.relative_to(ROOT)
out.mkdir(parents=True, exist_ok=True)
if (out / "supervisor.json").exists():
    raise SystemExit("Choose a new output directory; existing trial data will not be overwritten")
container_out = "/workspace/" + str(relative)
config_path = (ROOT / args.config).resolve()
shutil.copy2(config_path, out / "input_ppo.yaml")
profile = yaml.safe_load(config_path.read_text())
assert profile.get("garmi_task", {}).get("kitchen"), "This export supports the kitchen task"
evaluation_cases = profile.get("evaluation", {}).get("cases")
evaluation_count = (len(json.loads((ROOT / Path(evaluation_cases).relative_to("/workspace")).read_text()))
                    if evaluation_cases else args.eval_envs)
paired_count = (len(json.loads((ROOT / Path(args.paired_eval_cases).relative_to("/workspace")).read_text()))
                if args.paired_eval_cases else 0)
evaluation_schedule = ("在第 " + "、".join(map(str, sorted(set(args.eval_updates + [args.updates])))) + " 次更新"
                       if args.eval_updates else f"每 {args.eval_interval} 次更新及训练结束时")


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def host_path(path):
    return ROOT / Path(path).relative_to("/workspace")


def read_progress():
    path = out / "progress.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "starting", "update": 0}


def write_report(progress, supervision):
    rows = []
    for ev in progress.get("evaluations", []):
        counts = ev["counts"]
        link = os.path.relpath(host_path(ev["summary"]), out)
        rows.append(f'| [{ev["update"]}]({link}) | {ev["success_rate"]:.1%} | '
                    f'{counts.get("collision",0)} | {counts.get("dropped",0)} | '
                    f'{counts.get("out_of_bounds",0)} | {counts.get("attempt_failed",0)} | {counts.get("timeout",0)} |')
    status = supervision.get("status", progress["status"])
    if status == "running":
        status = progress["status"]
    task_path = out / "task.json"
    task = json.loads(task_path.read_text()) if task_path.exists() else {}
    fixed_finish = task.get("variant") == "fixed_finish_v3"
    control_description = ("RL 控制接近时的底盘和右臂，闭合、提起与保持由固定程序执行；"
                           "成功仍按实际物理接触与稳定抬升判定。" if fixed_finish else
                           "录像中的动作由学习策略输出。")
    progress_caption=(f'{progress.get("update",0)} 次更新；时间上限 {args.wall_hours} 小时' if args.wall_hours
                      else f'{progress.get("update",0)} / {args.updates} 次更新')
    text = f'''# {args.title}

状态：**{status}**；**{progress_caption}**。

{args.num_envs} 个并行环境，每轮采样 24 步；使用本目录保存的配置，运行期间不改参数。
[输入配置](input_ppo.yaml) · [实际任务与奖励参数](task.json) · [奖励项顺序](reward_terms.json)。
{evaluation_schedule}保存检查点并在独立进程评估固定的 {evaluation_count} 个起点，评估并行数 {args.eval_envs}，评估种子 12345。
训练种子 42。不会因评估失败率高而自动更改奖励或动作参数。
{control_description}

训练中已结束回合：{json.dumps(progress.get("episode_counts", {}), ensure_ascii=False)}。
训练回合统计包含随机探索；下表是使用策略均值动作的独立评估，两者不能混为同一成功率。

| PPO 更新 | 独立评估成功率 | 碰撞 | 掉落 | 出界 | 固定抓取未成功 | 超时 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

[每步耗时与诊断](steps.csv) · [PPO 更新与损失](updates.csv) · [逐回合结束原因](episodes.csv) · [运行日志](run.log) · [GPU 记录](gpu.csv)

[实时状态](progress.json) · [最近检查点](latest.json) · [最佳评估](best.json) · [配置](ppo.yaml) · [代码和资产哈希](fingerprint.json)

续训会恢复网络、优化器、标准化统计、学习率调度与随机数状态，从新的仿真回合开始；不宣称逐物理步重现中断前的场景。
'''
    if supervision.get("errors"):
        text += "\n监督进程检测到错误：" + json.dumps(supervision["errors"], ensure_ascii=False) + "\n"
    if args.paired_eval_cases:
        text += f"\n相同检查点另测固定 {paired_count} 场景，供模仿学习基线对照；两组测试分别记录。\n"
        text += f"\n| 更新 | {paired_count} 场景成功率 | 平均总回报 | 平均折扣回报 |\n|---:|---:|---:|---:|\n"
        baseline = out / "comparison-baseline.json"
        if baseline.exists():
            for row in json.loads(baseline.read_text())["paired"]:
                text += (f"| {row['label']} | {row['success_rate']:.1%} | "
                         f"{row['mean_return']:.3f} | {row['mean_discounted_return']:.3f} |\n")
        for row in progress.get("paired_evaluations", []):
            link = os.path.relpath(host_path(row["summary"]), out)
            text += (f"| [{row['update']}]({link}) | {row['success_rate']:.1%} | "
                     f"{row['mean_return']:.3f} | {row['mean_discounted_return']:.3f} |\n")
    if progress.get("evaluations") and "per_object" in progress["evaluations"][-1]:
        text += "\n最近一次训练参考验证：\n\n| 物体 | 成功率 | 结束原因 |\n| --- | ---: | --- |\n"
        for name, entry in progress["evaluations"][-1]["per_object"].items():
            text += f'| {name} | {entry["success_rate"]:.1%} | {json.dumps(entry["counts"],ensure_ascii=False)} |\n'
        if "heldout" in args.final_eval_splits:
            text += "\n留出的抓取参考只在训练结束后测试，不参与最佳检查点选择。\n"
        else:
            text += "\n本轮只测试配置中固定的抓取参考；最终测试换一批底盘起点，不测试未见抓取参考。\n"
    if args.wall_hours:
        text += f"\n本轮按实际运行时间约 {args.wall_hours} 小时停止（包括周期验证），更新次数是上限。\n"
    for evaluation in supervision.get("final_evaluations",[]):
        text += f'\n[最终测试：{evaluation["split"]}]({evaluation["path"]})，成功率 {evaluation["success_rate"]:.1%}。\n'
    for key in ("final_evaluation_errors", "video_errors"):
        if supervision.get(key):
            text += f'\n后处理未完成（{key}）：{json.dumps(supervision[key],ensure_ascii=False)}。训练检查点仍已保存。\n'
    for video in supervision.get("videos", []):
        text += f'\n[策略回放：{video["label"]}]({video["path"]})\n'
    (out / "REPORT.md").write_text(text)
    video_html = "".join(f'<p>{html.escape(v["label"])}</p><video controls src="{html.escape(v["path"])}" width="720"></video>'
                         for v in supervision.get("videos", []))
    table_rows = "".join(f'<tr><td>{ev["update"]}</td><td>{ev["success_rate"]:.1%}</td>'
                         f'<td>{html.escape(json.dumps(ev["counts"], ensure_ascii=False))}</td></tr>'
                         for ev in progress.get("evaluations", []))
    paired_html = ""
    if args.paired_eval_cases:
        paired_rows = []
        baseline = out / "comparison-baseline.json"
        if baseline.exists():
            paired_rows.extend(json.loads(baseline.read_text())["paired"])
        paired_rows.extend(dict(row, label=str(row["update"])) for row in progress.get("paired_evaluations", []))
        paired_html = (f'<p>固定 {paired_count} 场景，与模仿学习基线比较：</p>'
            '<table><tr><th>模型/更新</th><th>成功率</th><th>平均总回报</th><th>平均折扣回报</th></tr>'
            + ''.join(f'<tr><td>{html.escape(row["label"])}</td><td>{row["success_rate"]:.1%}</td>'
                      f'<td>{row["mean_return"]:.3f}</td><td>{row["mean_discounted_return"]:.3f}</td></tr>'
                      for row in paired_rows) + '</table>')
    object_rows = ""
    if progress.get("evaluations"):
        for name, entry in progress["evaluations"][-1].get("per_object", {}).items():
            object_rows += (f'<tr><td>{html.escape(name)}</td><td>{entry["success_rate"]:.1%}</td>'
                f'<td>{html.escape(json.dumps(entry["counts"],ensure_ascii=False))}</td></tr>')
    final_html = "".join(f'<p>最终测试 <a href="{html.escape(ev["path"])}">{ev["split"]}</a>：'
        f'{ev["success_rate"]:.1%}</p>' for ev in supervision.get("final_evaluations", []))
    (out / "index.html").write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>GARMI PPO 试训</title><style>body{{font:17px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#fafafa;color:#222}}
td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}video{{max-width:100%}}a{{color:#1768a6}}</style>
<h1>{html.escape(args.title)}</h1><p>状态：<b>{html.escape(status)}</b>；{progress_caption}</p>
<p>{args.num_envs} 个环境，使用冻结的试验配置；{evaluation_schedule}独立评估 {evaluation_count} 个固定起点。</p>
<p>重新载入页面可查看最新结果。{control_description}</p>
<table><tr><th>更新</th><th>固定测试集成功率</th><th>结束原因</th></tr>{table_rows}</table>
{paired_html}
<p>最近一次验证的分物体结果：</p><table><tr><th>物体</th><th>成功率</th><th>结束原因</th></tr>{object_rows}</table>
{final_html}
<p><a href="REPORT.md">完整记录</a> · <a href="steps.csv">每一步耗时</a> · <a href="updates.csv">PPO 损失</a> ·
<a href="run.log">运行日志</a> · <a href="progress.json">实时状态</a></p>{video_html}</html>''')


files = list((ROOT / "rl/garmi_rl").glob("*.py")) + list((ROOT / "rl").glob("*.py"))
files.append(config_path)
files += list((ROOT / "rl/assets/multi").glob("*.json"))
for p in files:
    destination = out / "source" / p.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, destination)
files += [p for p in (ROOT / "rl/assets").rglob("*") if p.suffix in (".usd", ".usda", ".json")]
atomic_json(out / "fingerprint.json", {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
cmd = CONTAINER + ["/isaac-sim/python.sh",
       "/workspace/rl/train_trial.py", "--headless", "--num_envs", str(args.num_envs),
       "--updates", str(args.updates), "--eval-interval", str(args.eval_interval),
       "--eval-envs", str(args.eval_envs), "--out", container_out,
       "--config", container_out+"/input_ppo.yaml"]
if args.resume:
    cmd += ["--resume", args.resume]
if args.allow_env_resize:
    cmd += ["--allow-env-resize"]
if args.defer_evaluation:
    cmd += ["--skip-evaluation"]
if args.init_checkpoint:
    cmd += ["--init-checkpoint", args.init_checkpoint]
if args.wall_hours:
    cmd += ["--wall-hours", str(args.wall_hours)]
if args.eval_updates:
    cmd += ["--eval-updates", *map(str, args.eval_updates)]
if args.paired_eval_cases:
    cmd += ["--paired-eval-cases", args.paired_eval_cases,
            "--paired-eval-envs", str(args.paired_eval_envs),
            "--paired-eval-seed", str(args.paired_eval_seed)]
supervision = dict(status="running", host_pid=os.getpid(), start_unix=time.time(), command=cmd,
                   errors=[], videos=[], final_evaluation_splits=args.final_eval_splits)
atomic_json(out / "supervisor.json", supervision)
stop = False


def request_stop(signum, frame):
    global stop
    stop = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
with (out / "run.log").open("w") as log, (out / "gpu.csv").open("w", buffering=1) as gpu:
    gpu.write("timestamp,memory_used_mib,utilization_percent,temperature_c\n")
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    supervision["container_exec_pid"] = process.pid
    atomic_json(out / "supervisor.json", supervision)
    sent_stop = False
    offset = 0
    tail = ""
    while process.poll() is None:
        progress = read_progress()
        with (out / "run.log").open(errors="replace") as source:
            source.seek(offset)
            chunk = source.read()
            offset = source.tell()
        tail = (tail + chunk)[-65536:]
        for word in ("Traceback", "CUDA error", "out of memory", "buffer overflow",
                     "increase PxgDynamicsMemoryConfig", "PhysX error"):
            if word.lower() in tail.lower() and word not in supervision["errors"]:
                supervision["errors"].append(word)
                stop = True
        if stop and not sent_stop and progress.get("pid"):
            subprocess.run(CONTAINER + ["kill", "-TERM", str(progress["pid"])], check=False)
            sent_stop = True
        sample = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if sample.returncode == 0:
            gpu.write(sample.stdout)
        atomic_json(out / "supervisor.json", supervision)
        write_report(progress, supervision)
        time.sleep(5)
    supervision["exit_code"] = process.returncode
progress = read_progress()
log_text = (out / "run.log").read_text(errors="replace")
for word in ("Traceback", "CUDA error", "out of memory", "buffer overflow", "PhysX error"):
    if word.lower() in log_text.lower() and word not in supervision["errors"]:
        supervision["errors"].append(word)
completed = process.returncode == 0 and not supervision["errors"] and progress["status"] == "complete" and "TRIAL_DONE" in log_text
supervision["status"] = "complete" if completed else ("interrupted" if progress["status"] == "interrupted" else "failed")

if completed and args.defer_evaluation and not stop:
    supervision["status"] = "running"
    deferred_command = CONTAINER + [
        "/isaac-sim/python.sh", "/workspace/rl/evaluate_trial_checkpoint.py",
        "--out", container_out, "--checkpoint", progress["latest_checkpoint"],
        "--eval-envs", str(args.eval_envs)]
    if args.paired_eval_cases:
        deferred_command += ["--paired-eval-cases", args.paired_eval_cases,
            "--paired-eval-envs", str(args.paired_eval_envs), "--paired-eval-seed", str(args.paired_eval_seed)]
    supervision["deferred_evaluation_command"] = deferred_command
    with (out / "deferred-evaluation.log").open("w") as log, (out / "gpu.csv").open("a", buffering=1) as gpu:
        deferred = subprocess.Popen(deferred_command, stdout=log, stderr=subprocess.STDOUT)
        sent_stop = False
        while deferred.poll() is None:
            progress = read_progress()
            if stop and not sent_stop and progress.get("evaluation_mode") == "after_training_process_exit":
                subprocess.run(CONTAINER + ["kill", "-TERM", str(progress["pid"])], check=False)
                sent_stop = True
            sample = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if sample.returncode == 0:
                gpu.write(sample.stdout)
            atomic_json(out / "supervisor.json", supervision)
            write_report(progress, supervision)
            time.sleep(5)
    progress = read_progress()
    supervision["deferred_evaluation_exit_code"] = deferred.returncode
    completed = (deferred.returncode == 0 and progress["status"] == "complete"
                 and "DEFERRED_EVALUATION_DONE" in (out / "deferred-evaluation.log").read_text())
    supervision["status"] = "complete" if completed else progress.get("status", "failed")
    if not completed:
        supervision["errors"].append("Deferred evaluation failed or interrupted")

# Kitchen recordings use evaluate_kitchen.py after the training process exits.
supervision["end_unix"] = time.time()
atomic_json(out / "supervisor.json", supervision)
write_report(progress, supervision)
subprocess.run(CONTAINER + ["chown", "-R", f"{os.getuid()}:{os.getgid()}", container_out], check=False)
print(json.dumps(supervision, ensure_ascii=False), flush=True)
