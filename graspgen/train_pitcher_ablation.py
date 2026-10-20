"""Prepare and run three isolated 100-update pitcher PPO comparisons on GPU 0."""
import argparse
import collections
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/pitcher-ppo-ablation-v1"
CODE = OUT / "code/rl"
BC = ROOT / "runs/wrist-imitation-v6/pitcher/round-03-bc/last.pt"
ARMS = ("reward-only", "noise-only", "reward-and-noise")


def read(p):
    return json.loads(p.read_text())


def write(p, data):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inside(p):
    return "/workspace/" + str(p.relative_to(ROOT))


def replace_once(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (path, old)
    path.write_text(text.replace(old, new))


def verify_protected():
    mug = read(ROOT / "runs/wrist-ppo-v6/mug/supervisor.json")
    plan = read(ROOT / mug["active_plan"])
    for section in ("source_sha256", "code_sha256"):
        for rel, digest in plan[section].items():
            assert sha(ROOT / rel) == digest, rel
    pitcher = read(ROOT / "runs/wrist-ppo-v6/pitcher/supervisor.json")
    assert pitcher["status"] == "paused" and pitcher["ppo_updates"] == 1873
    return dict(mug_status=mug["status"], mug_plan=mug["active_plan"],
                original_pitcher_status="paused", original_pitcher_update=1873,
                all_original_frozen_hashes_match=True)


def prepare():
    assert not OUT.exists(), "Use a new revision instead of overwriting an experiment"
    protected = verify_protected()
    CODE.mkdir(parents=True)
    source_hashes = {}
    for path in (ROOT / "rl").rglob("*.py"):
        relative = path.relative_to(ROOT / "rl")
        target = CODE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
    # These are read-only shared assets, not a second copy of the dataset.
    (CODE / "assets").symlink_to(os.path.relpath(ROOT / "rl/assets", CODE))
    for name in ("assets", "resources", "runs"):
        (CODE.parent / name).symlink_to(os.path.relpath(ROOT / name, CODE.parent))
    shutil.copy2(ROOT / "graspgen/pitcher_precision_reward.py", CODE / "garmi_rl/precision_reward.py")
    shutil.copy2(ROOT / "graspgen/pitcher_ablation_audit.py", CODE / "ablation_audit.py")
    replace_once(CODE / "garmi_rl/config.py",
        '        elif approach_reward != "legacy":',
        '''        elif approach_reward == "pitcher_precision_v1":
            from . import precision_reward
            parameters = task["approach_reward_parameters"]
            assert set(parameters) == {"coarse_weight", "position_sigma_m",
                "angle_sigma_rad", "distance_clip_m", "distance_scale_per_m"}
            cfg.rewards.approach.func = precision_reward.policy_approach
            cfg.rewards.approach.params = dict(parameters)
            description["approach_reward"] = precision_reward.description(parameters)
        elif approach_reward != "legacy":''')
    replace_once(CODE / "train_trial.py",
        '    print("TRANSFER_LOADED",args.init_checkpoint,"fresh optimizer and scheduler",flush=True)',
        '''    scale = float(agent_cfg["ablation"]["initial_std_scale"])
    assert scale in (1., .5)
    before = agent.policy.log_std_parameter.detach().clone()
    with torch.no_grad():
        agent.policy.log_std_parameter.add_(float(np.log(scale)))
    for name in ("policy", "value"):
        for key, value in agent.checkpoint_modules[name].state_dict().items():
            expected = weights[name][key]
            if key == "log_std_parameter":
                expected = expected + float(np.log(scale))
            assert torch.equal(value, expected), (name, key)
    import garmi_rl.config as config_module
    assert Path(config_module.__file__).resolve().parent.parent == Path(__file__).resolve().parent
    atomic_json(out / "initial_noise.json", dict(scale=scale, before=before.exp().tolist(),
        after=agent.policy.log_std_parameter.detach().exp().tolist(),
        non_noise_weights_identical=True, module=str(config_module.__file__)))
    print("TRANSFER_LOADED",args.init_checkpoint,"fresh optimizer and scheduler",flush=True)''')
    # This only corrects the diagnostic counter; the controller was already strict.
    replace_once(CODE / "train_trial.py",
        'aligned_count=int(((d["distance"] < .015) & (d["angle"] < .15)).sum()),',
        'aligned_count=int(((d["distance"] < .005) & (d["angle"] < .05)).sum()),')
    # Save complete rollout traces at each scheduled validation, including phase transitions.
    replace_once(CODE / "garmi_rl/trial_evaluation.py",
        '"--checkpoint", str(checkpoint), "--config", str(out / "ppo.yaml"),\n           "--out", str(destination)]',
        '"--checkpoint", str(checkpoint), "--config", str(out / "ppo.yaml"),\n           "--out", str(destination), "--save-rollouts"]')

    sys.path.insert(0, str(ROOT / "rl"))
    from garmi_rl.imitation_pipeline import validate_partitions, case_key
    data = ROOT / "runs/wrist-expert-collection-v6/pitcher"
    train, val = read(data / "train-index.json"), read(data / "validation-index.json")
    heldout = read(ROOT / "runs/wrist-ppo-v6/pitcher/evaluation-cases.json")
    refs = validate_partitions(train, val, heldout, "pitcher")
    rng = random.Random(2026102007)
    pools = collections.defaultdict(list)
    for row in val["demonstrations"]:
        pools[row["reference"]].append(row)
    selected = [row for ref in refs for row in rng.sample(pools[ref], 2)]
    cases = [row["initial_case"] for row in selected]
    excluded = {case_key(row["initial_case"]) for row in train["demonstrations"]}
    excluded.update(case_key(c) for c in heldout)
    assert len(cases) == 80 and not any(case_key(c) in excluded for c in cases)
    write(OUT / "validation-cases.json", cases)
    write(OUT / "validation-selection.json", dict(split="validation", demonstrations=selected))
    write(OUT / "split-audit.json", dict(status="passed", cases=80, references=40,
        per_reference=2, train_cases_excluded=12000, heldout_cases_excluded=80,
        source_validation_index_sha256=sha(data / "validation-index.json")))
    original = yaml.safe_load((ROOT / "runs/wrist-ppo-v6/pitcher/ppo.yaml").read_text())
    for arm in ARMS:
        cfg = copy.deepcopy(original)
        cfg["evaluation"] = dict(cases=inside(OUT / "validation-cases.json"),
                                cases_sha256=sha(OUT / "validation-cases.json"))
        cfg["ablation"] = dict(initial_std_scale=1. if arm == "reward-only" else .5)
        if arm != "noise-only":
            cfg["garmi_task"]["approach_reward"] = "pitcher_precision_v1"
            cfg["garmi_task"]["approach_reward_parameters"] = dict(
                coarse_weight=.2, position_sigma_m=.02, angle_sigma_rad=.10,
                distance_clip_m=2., distance_scale_per_m=2.)
        (OUT / (arm + ".yaml")).write_text(yaml.safe_dump(cfg, sort_keys=False))
    frozen_paths = list(CODE.rglob("*.py")) + list(OUT.glob("*.yaml")) + [
        OUT / "validation-cases.json", OUT / "validation-selection.json", OUT / "split-audit.json",
        BC, ROOT / "runs/wrist-imitation-v6/pitcher/round-03-train-index.json",
        ROOT / "runs/wrist-ppo-v6/pitcher/training-reset-pool.json",
        Path(__file__).resolve()]
    write(OUT / "plan.json", dict(status="prepared", arms=list(ARMS), updates=100,
        evaluation_updates=[25, 50, 75, 100], num_envs=4096, gpu="0",
        seed=2026102001, validation_seed=2026102007, bc_checkpoint=inside(BC),
        bc_sha256=sha(BC), evaluation="80 validation cases, not the frozen heldout test set",
        limitations=["Single training seed; screening experiment, not a definitive causal estimate",
                     "No simultaneous unchanged-PPO control; original run provides historical context",
                     "arm_target_clip_fraction remains a legacy counter; do not interpret it"],
        protected=protected, original_code_sha256=source_hashes,
        frozen_sha256={str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}))
    write(OUT / "supervisor.json", dict(status="prepared", stages=[]))
    print("ABLATION_PREPARED", OUT, flush=True)


def resources():
    result = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used,memory.total",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    used, total = map(int, result.stdout.strip().split(","))
    ram = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("MemAvailable:"))) / 1024
    return dict(gpu_used_mib=used, gpu_total_mib=total, ram_available_mib=ram)


def run():
    lock = (OUT / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan, state = read(OUT / "plan.json"), read(OUT / "supervisor.json")
    assert state["status"] == "prepared"
    state.update(status="running", pid=os.getpid(), gpu="0", started_unix=time.time())
    write(OUT / "supervisor.json", state)

    def stage(name, script, args, marker, training=False):
        verify_protected()
        for rel, digest in plan["frozen_sha256"].items():
            assert sha(ROOT / rel) == digest, rel
        if training:
            while True:
                r = resources()
                # 12 GiB training + ~5 GiB independent validation + 4 GiB reserve.
                if r["gpu_total_mib"] - r["gpu_used_mib"] >= 21504 and r["ram_available_mib"] >= 32768:
                    break
                state.update(status="waiting_resources", next_stage=name, resources=r)
                write(OUT / "supervisor.json", state)
                time.sleep(30)
        command = ["bash", "scripts/rl.sh", inside(CODE / script)] + list(map(str, args))
        row = dict(name=name, status="running", started_unix=time.time(), command=command)
        state["stages"].append(row)
        state.update(status="running", current_stage=name)
        write(OUT / "supervisor.json", state)
        with (OUT / (name + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, GARMI_RL_GPU="0"),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = child.pid
            write(OUT / "supervisor.json", state)
            while child.poll() is None:
                state["updated_unix"] = time.time()
                write(OUT / "supervisor.json", state)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        text = (OUT / (name + ".log")).read_text(errors="replace")
        bad = [s for s in ("Traceback (most recent call last)", "CUDA error:", "out of memory",
                           "buffer overflow", "PhysX error", "increase PxgDynamicsMemoryConfig")
               if s.lower() in text.lower()]
        row.update(returncode=child.returncode, errors=bad, finished_unix=time.time(),
                   status="complete" if child.returncode == 0 and marker in text and not bad else "failed")
        write(OUT / "supervisor.json", state)
        assert row["status"] == "complete", (name, bad)
        print("ABLATION_STAGE_DONE", name, flush=True)

    try:
        stage("preflight", "ablation_audit.py",
              ["--root", inside(OUT), "--out", inside(OUT / "bc-label-audit.json")],
              "ABLATION_AUDIT_PASSED")
        stage("reset-check", "check_local_ppo.py",
              ["--headless", "--config", inside(OUT / "reward-only.yaml"),
               "--out", inside(OUT / "reset-check"), "--num_envs", 256],
              "LOCAL_PPO_RESET_PASSED")
        assert read(OUT / "reset-check/summary.json")["status"] == "passed"
        stage("bc-validation", "evaluate_kitchen.py", [
            "--headless", "--config", inside(OUT / "noise-only.yaml"),
            "--checkpoint", inside(BC), "--out", inside(OUT / "bc-validation"),
            "--num_envs", 80, "--seed", 12345, "--require-strict-success", "--save-rollouts"],
            "EVALUATION_DONE")
        baseline = read(OUT / "bc-validation/summary.json")
        assert baseline["completed"] == 80 and baseline["require_strict_success"]
        assert baseline["evaluation_cases_sha256"] == sha(OUT / "validation-cases.json")
        assert baseline["checkpoint_sha256"] == plan["bc_sha256"]
        comparisons = dict(status="running", bc_validation=dict(
            success_rate=baseline["success_rate"], counts=baseline["counts"]), arms={})
        write(OUT / "comparison.json", comparisons)
        for arm in ARMS:
            dest = OUT / arm
            stage(arm, "train_trial.py", [
                "--headless", "--config", inside(OUT / (arm + ".yaml")),
                "--out", inside(dest), "--num_envs", 4096, "--updates", 100,
                "--eval-updates", 25, 50, 75, 100, "--eval-envs", 80,
                "--seed", plan["seed"], "--eval-seed", 12345, "--init-checkpoint", inside(BC)],
                "TRIAL_DONE", training=True)
            summary = read(dest / "summary.json")
            assert summary["status"] == "complete" and summary["update"] == 100
            assert summary["frozen_state_normalization_verified"]
            rows = []
            for update in plan["evaluation_updates"]:
                checkpoint = dest / f"checkpoints/update_{update:04d}.pt"
                audit_path = dest / f"validation-label-audit-{update:04d}.json"
                stage(f"{arm}-labels-{update:04d}", "ablation_audit.py",
                      ["--root", inside(OUT), "--checkpoint", inside(checkpoint),
                       "--out", inside(audit_path)], "ABLATION_AUDIT_PASSED")
                evaluation = read(dest / f"evaluations/update_{update:04d}/summary.json")
                assert evaluation["completed"] == 80 and evaluation["require_strict_success"]
                assert evaluation["evaluation_cases_sha256"] == sha(OUT / "validation-cases.json")
                assert evaluation["checkpoint_sha256"] == sha(checkpoint)
                assert evaluation["initial_cases"] == baseline["initial_cases"]
                assert all(r["strict_success_verified"] for r in evaluation["results"]
                           if r["reason"] == "success")
                rows.append(dict(update=update, success_rate=evaluation["success_rate"],
                    counts=evaluation["counts"],
                    handovers=sum(r["final_phase"] > 0 for r in evaluation["results"]),
                    expert_action_mse=read(audit_path)["expert_action_mse"]))
            stage(f"{arm}-transfer-audit", "audit_ppo_transfer.py",
                  ["--bc", inside(BC), "--checkpoint", inside(dest / "checkpoints/update_0100.pt"),
                   "--stage", inside(dest), "--expected-update", 100,
                   "--out", inside(dest / "transfer-audit.json")], "PPO_TRANSFER_AUDIT_PASSED")
            comparisons["arms"][arm] = rows
            write(OUT / "comparison.json", comparisons)
        comparisons["status"] = "complete"
        write(OUT / "comparison.json", comparisons)
        state.update(status="complete", protected=verify_protected())
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["updated_unix"] = time.time()
        write(OUT / "supervisor.json", state)
        print("ABLATION_SUPERVISOR", state["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else run()
