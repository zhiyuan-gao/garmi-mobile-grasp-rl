"""Shared fixed-case evaluations, usable with or without a live training process."""
import collections
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def run_evaluator(command, log):
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=1230))
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        raise


def evaluate_checkpoint(checkpoint, update, *, out, args, agent_cfg, progress):
    destination = out / "evaluations" / f"update_{update:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    progress["status"] = "evaluating"
    atomic_json(out / "progress.json", progress)
    evaluator = "evaluate_kitchen.py" if "kitchen" in agent_cfg.get("garmi_task", {}) else "evaluate.py"
    cmd = ["timeout", "--kill-after=15", "1200", "/isaac-sim/python.sh", str(Path(__file__).resolve().parents[1] / evaluator),
           "--headless", "--num_envs", str(args.eval_envs), "--seed", str(args.eval_seed),
           "--checkpoint", str(checkpoint), "--config", str(out / "ppo.yaml"),
           "--out", str(destination)]
    tick = time.perf_counter()
    with (destination / "run.log").open("w") as log:
        result = run_evaluator(cmd, log)
    log_text = (destination / "run.log").read_text(errors="replace")
    bad = [word for word in ("Traceback", "CUDA error", "out of memory", "buffer overflow",
                             "increase PxgDynamicsMemoryConfig", "PhysX error")
           if word.lower() in log_text.lower()]
    if result.returncode or bad or "EVALUATION_DONE" not in log_text:
        raise RuntimeError(f"Independent evaluation failed: returncode={result.returncode}, errors={bad}")
    summary = json.loads((destination / "summary.json").read_text())
    if any(r is None for r in summary["results"]):
        raise RuntimeError("Evaluation did not finish all episodes")
    initial_hash = hashlib.sha256(json.dumps(summary.get("initial_cases",summary["initial_states"])).encode()).hexdigest()
    if progress.get("eval_initial_states_sha256", initial_hash) != initial_hash:
        raise RuntimeError("Evaluation starting states changed between checkpoints")
    progress["eval_initial_states_sha256"] = initial_hash
    entry = dict(update=update, success_rate=summary["success_rate"],
                 counts=dict(collections.Counter(r["reason"] for r in summary["results"])),
                 mean_episode_seconds=statistics.mean(r["seconds"] for r in summary["results"]),
                 wall_seconds=time.perf_counter()-tick,
                 summary=str(destination / "summary.json"), checkpoint=str(checkpoint))
    if "per_object" in summary:
        entry.update(per_object=summary["per_object"],per_reference=summary["per_reference"])
    progress["evaluations"].append(entry)
    best = max(progress["evaluations"], key=lambda r: (r["success_rate"], r["update"]))
    progress["best_evaluation"] = best
    atomic_json(out / "best.json", best)
    progress["status"] = "training"
    print("EVALUATION", json.dumps(entry), flush=True)


def evaluate_paired_checkpoint(checkpoint, update, *, out, args, agent_cfg, progress):
    """Evaluate the unchanged fixed paired cases, independently of training."""
    cases_path = Path(args.paired_eval_cases)
    case_hash = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    expected_count = len(json.loads(cases_path.read_text()))
    if progress.get("paired_case_sha256", case_hash) != case_hash:
        raise RuntimeError("Paired evaluation cases changed between checkpoints")
    destination = out / "paired-evaluations" / f"update_{update:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    progress.update(status="evaluating_paired", paired_case_sha256=case_hash)
    atomic_json(out / "progress.json", progress)
    command = ["timeout", "--kill-after=15", "1200", "/isaac-sim/python.sh",
        str(Path(__file__).resolve().parents[1] / "evaluate_kitchen.py"), "--headless",
        "--num_envs", str(args.paired_eval_envs), "--seed", str(args.paired_eval_seed),
        "--checkpoint", str(checkpoint), "--config", str(out / "ppo.yaml"),
        "--cases", str(cases_path), "--out", str(destination)]
    tick = time.perf_counter()
    with (destination / "run.log").open("w") as log:
        result = run_evaluator(command, log)
    log_text = (destination / "run.log").read_text(errors="replace")
    bad = [word for word in ("Traceback", "CUDA error", "out of memory", "buffer overflow",
        "increase PxgDynamicsMemoryConfig", "PhysX error") if word.lower() in log_text.lower()]
    if result.returncode or bad or "EVALUATION_DONE" not in log_text:
        raise RuntimeError(f"Paired evaluation failed: returncode={result.returncode}, errors={bad}")
    summary = json.loads((destination / "summary.json").read_text())
    if (summary["status"] != "complete" or summary["completed"] != expected_count
            or summary["evaluation_cases_sha256"] != case_hash
            or summary["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
        raise RuntimeError("Paired evaluation did not finish or used mismatched inputs")
    # The trace mean is over active episodes before terminal episodes are removed.
    # Recover cohort returns without adding GPU work to policy evaluation.
    total = discounted = 0.
    gamma = agent_cfg["agent"]["discount_factor"]
    for line in (destination / "trace.jsonl").read_text().splitlines():
        row = json.loads(line)
        reward_sum = row["active"] * row["mean_reward"]
        total += reward_sum
        discounted += gamma ** row["step"] * reward_sum
    entry = dict(update=update, episodes=expected_count, success_rate=summary["success_rate"],
        counts=summary["counts"], per_reference=summary["per_reference"],
        mean_return=total/expected_count, mean_discounted_return=discounted/expected_count,
        wall_seconds=time.perf_counter()-tick, summary=str(destination / "summary.json"),
        checkpoint=str(checkpoint), cases_sha256=case_hash)
    progress.setdefault("paired_evaluations", []).append(entry)
    progress["status"] = "training"
    print("PAIRED_EVALUATION", json.dumps(entry), flush=True)

