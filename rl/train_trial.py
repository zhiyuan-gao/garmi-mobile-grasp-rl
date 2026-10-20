"""PPO trial with durable checkpoints, fixed independent evaluation and raw timings.

The interaction/update order follows skrl 1.4.3 SequentialTrainer. Task and
learning settings come from the selected frozen PPO/task profile.
"""
import argparse
import time
from pathlib import Path

started = time.perf_counter()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--updates", type=int, default=1000)
parser.add_argument("--eval-interval", type=int, default=100)
parser.add_argument("--eval-updates", type=int, nargs="+", help="Explicit evaluation updates, replacing the interval; final update is always evaluated")
parser.add_argument("--eval-envs", type=int, default=100)
parser.add_argument("--paired-eval-cases", help="Optional second fixed case set, evaluated at the same checkpoints")
parser.add_argument("--paired-eval-envs", type=int, default=120)
parser.add_argument("--paired-eval-seed", type=int, default=20261018)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval-seed", type=int, default=12345)
parser.add_argument("--out", required=True)
parser.add_argument("--config", default=str(Path(__file__).parent / "ppo.yaml"))
parser.add_argument("--resume", help="Checkpoint from this trial runner; starts fresh episodes")
parser.add_argument("--allow-env-resize", action="store_true",
                    help="Explicitly allow a new environment count on resume; preserve optimizer and cumulative samples")
parser.add_argument("--init-checkpoint", help="Transfer policy/value and normalizers into a new task; fresh optimizer")
parser.add_argument('--extend-expert-data',action='store_true',help='On resume only, allow an audited append-only extension of the expert dataset')
parser.add_argument("--wall-hours",type=float,default=0.,help="Wall-clock budget including periodic evaluation")
parser.add_argument("--skip-evaluation", action="store_true", help="Capacity checks only; still save checkpoints")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if min(args.num_envs, args.updates, args.eval_interval, args.eval_envs, args.paired_eval_envs) < 1:
    parser.error("Environment/update counts and evaluation interval must be positive")
if args.eval_updates and any(u < 1 or u > args.updates for u in args.eval_updates):
    parser.error("Evaluation updates must be within the training budget")
evaluation_updates = set(args.eval_updates) if args.eval_updates else None
if args.extend_expert_data and not args.resume:
    parser.error('--extend-expert-data requires --resume')
if args.allow_env_resize and not args.resume:
    parser.error('--allow-env-resize requires --resume')
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if (out / "progress.json").exists() and not args.resume:
    parser.error("Output already contains a run; choose a new directory or --resume")
app = AppLauncher(args).app

import collections
import copy
import csv
import hashlib
import json
import os
import random
import signal
import subprocess
import traceback
import numpy as np
import torch
import yaml
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from garmi_rl.expert_learning import ExpertRunner as Runner
from garmi_rl.config import configured_env


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


class CsvLog:
    def __init__(self, name):
        path = out / name
        self.has_header = path.exists() and path.stat().st_size > 0
        self.fieldnames = next(csv.reader(path.open())) if self.has_header else None
        self.file = path.open("a", buffering=1)
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames or list(row), extrasaction="ignore")
            if not self.has_header:
                self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.file.close()


agent_cfg = yaml.safe_load(Path(args.config).read_text())
cfg, task_description = configured_env(agent_cfg, args.num_envs, args.seed)
fixed_finish = task_description["variant"] == "fixed_finish_v3"
terminal_names = ("collision", "dropped", "out_of_bounds", "success") + (("attempt_failed",) if fixed_finish else ())
env = ManagerBasedRLEnv(cfg=cfg)
env.forbidden_sensors = cfg.forbidden_sensors
wrapped = SkrlVecEnvWrapper(env)
rollout = agent_cfg["agent"]["rollouts"]
total_steps = args.updates * rollout
agent_cfg["seed"] = args.seed
agent_cfg["trainer"]["timesteps"] = total_steps
agent_cfg["agent"]["experiment"].update(
    directory=str(out), experiment_name=f"training_{time.time_ns()}", checkpoint_interval=0)
(out / "ppo.yaml").write_text(yaml.safe_dump(agent_cfg))
atomic_json(out / "task.json", task_description)
atomic_json(out / "reward_terms.json", list(env.reward_manager.active_terms))
runner = Runner(wrapped, agent_cfg)
agent = runner.agent
if agent_cfg.get('expert_bc'):
    atomic_json(out / 'expert_data.json', agent.expert_metadata)
agent.set_running_mode("train")
if args.init_checkpoint:
    if args.resume:
        raise ValueError("Choose resume or initialization, not both")
    weights=torch.load(args.init_checkpoint,map_location=env.device,weights_only=False)
    from garmi_rl.joint_delta import ACTION_SPACE, is_joint_delta
    if is_joint_delta(agent_cfg) != (weights.get('action_contract',{}).get('name') == ACTION_SPACE):
        raise ValueError('Initialization checkpoint and environment action semantics differ')
    for name in ("policy","value","state_preprocessor","value_preprocessor"):
        agent.checkpoint_modules[name].load_state_dict(weights[name])
    print("TRANSFER_LOADED",args.init_checkpoint,"fresh optimizer and scheduler",flush=True)
step_log, update_log, episode_log = [CsvLog(n) for n in ("steps.csv", "updates.csv", "episodes.csv")]
checkpoint_dir = out / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
stop_requested = False


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True
    print("STOP_REQUESTED: will save after the current PPO rollout", flush=True)


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
progress = dict(status="initializing", update=0, target_updates=args.updates,
                num_envs=args.num_envs, rollout=rollout, seed=args.seed,
                eval_envs=args.eval_envs, eval_seed=args.eval_seed,
                episode_counts={}, evaluations=[], segments=[])
progress["initialization_checkpoint"]=args.init_checkpoint
progress["wall_hours"]=args.wall_hours
resume_update = 0
resume_transitions = 0
resume_envs = args.num_envs
if args.resume:
    checkpoint = Path(args.resume)
    resume_state = torch.load(checkpoint.with_suffix(".resume.pt"), map_location="cpu", weights_only=False)
    resume_envs = resume_state["num_envs"]
    if resume_state["rollout"] != rollout:
        raise ValueError("Resume must keep rollout unchanged")
    if resume_envs != args.num_envs and not args.allow_env_resize:
        raise ValueError("Changing num_envs on resume requires --allow-env-resize")
    saved_cfg = resume_state["learning_config"]
    current_cfg = {"models": agent_cfg["models"], "agent": {
        k: v for k, v in agent_cfg["agent"].items() if k != "experiment"}}
    if "garmi_task" in agent_cfg:
        current_cfg["garmi_task"] = agent_cfg["garmi_task"]
    if "expert_bc" in agent_cfg:
        current_cfg["expert_bc"] = agent_cfg["expert_bc"]
    compatible_cfg=copy.deepcopy(current_cfg)
    if args.extend_expert_data:
        from garmi_rl.expert_learning import check_expert_extension
        previous_index=saved_cfg['expert_bc']['index']
        if resume_state.get('expert_index_sha256'):
            assert hashlib.sha256(Path(previous_index).read_bytes()).hexdigest()==resume_state['expert_index_sha256']
        extension=check_expert_extension(previous_index,current_cfg['expert_bc']['index'])
        compatible_cfg['expert_bc']['index']=previous_index
        atomic_json(out/'expert_data_extension.json',extension)
    if saved_cfg != compatible_cfg:
        raise ValueError("Resume learning configuration differs from the checkpoint")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != resume_state["checkpoint_sha256"]:
        raise ValueError("Checkpoint and resume metadata do not match")
    agent.load(str(checkpoint))
    agent.scheduler.load_state_dict(resume_state["scheduler"])
    agent.scaler.load_state_dict(resume_state["scaler"])
    torch.set_rng_state(resume_state["torch_rng"])
    torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
    np.random.set_state(resume_state["numpy_rng"])
    random.setstate(resume_state["python_rng"])
    resume_update = resume_state["update"]
    progress = resume_state["progress"]
    resume_transitions = progress.get("transitions", resume_update * rollout * resume_envs)
    if resume_envs != args.num_envs:
        resize = dict(update=resume_update, from_num_envs=resume_envs, to_num_envs=args.num_envs,
                      transitions=resume_transitions, checkpoint=str(checkpoint))
        progress.setdefault("environment_resizes", []).append(resize)
        atomic_json(out / f"environment_resize_{resume_update:04d}.json", resize)
        print("RESUME_ENV_RESIZE", json.dumps(resize), flush=True)
    progress.update(status="resuming", target_updates=args.updates, num_envs=args.num_envs,
                    transitions=resume_transitions, latest_checkpoint=str(checkpoint))
    print(f"RESUME_LOADED update={resume_update}; optimizer/scalers restored; fresh episodes", flush=True)
if resume_update >= args.updates:
    raise ValueError("Target updates must exceed the checkpoint update")
progress["evaluation_updates"] = sorted(evaluation_updates) if evaluation_updates is not None else None

initialization = dict(mode="resume" if args.resume else "transfer" if args.init_checkpoint else "scratch",
    checkpoint=args.resume or args.init_checkpoint, seed=args.seed,
    optimizer_state_entries=len(agent.optimizer.state), normalizers={},
    previous_num_envs=resume_envs, num_envs=args.num_envs, transitions=resume_transitions,
    optimizer_steps=sorted({float(s["step"]) for s in agent.optimizer.state.values() if "step" in s}),
    learning_rates=[g["lr"] for g in agent.optimizer.param_groups])
for name in ("state_preprocessor", "value_preprocessor"):
    scaler = agent.checkpoint_modules.get(name)
    if scaler is not None and hasattr(scaler, "current_count"):
        initialization["normalizers"][name] = dict(count=float(scaler.current_count),
            mean_max_abs=float(scaler.running_mean.abs().max()),
            variance_min=float(scaler.running_variance.min()),
            variance_max=float(scaler.running_variance.max()))
if initialization["mode"] == "scratch":
    assert initialization["optimizer_state_entries"] == 0
    assert all(s["count"] == 1. and s["mean_max_abs"] == 0.
               and s["variance_min"] == s["variance_max"] == 1.
               for s in initialization["normalizers"].values())
atomic_json(out / f"initialization_{resume_update:04d}.json", initialization)
print("INITIALIZATION_CHECK", json.dumps(initialization), flush=True)
frozen_state_normalization = (copy.deepcopy(agent.checkpoint_modules['state_preprocessor'].state_dict())
                              if agent_cfg.get('expert_bc') else None)

counts = collections.Counter(progress["episode_counts"])
segment_id = len(progress["segments"])
segment = dict(id=segment_id, start_update=resume_update, start_unix=time.time(),
               startup_seconds=time.perf_counter()-started, resume=args.resume,
               num_envs=args.num_envs, start_transitions=resume_transitions)
progress["segments"].append(segment)
progress["pid"] = os.getpid()
progress["status"] = "training"
atomic_json(out / "progress.json", progress)
obs, _ = wrapped.reset()
episode_return = torch.zeros(args.num_envs, device=env.device)
episode_length = torch.zeros(args.num_envs, device=env.device, dtype=torch.long)
step_times = []
update_times = []
last_update_metrics = {}
original_update = agent._update


def measured_update(*a, **kw):
    global last_update_metrics
    torch.cuda.synchronize()
    tick = time.perf_counter()
    original_update(*a, **kw)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - tick
    metrics = {k: float(v[-1]) for k, v in agent.tracking_data.items()
               if v and k.startswith(("Loss", "Learning", "Policy"))}
    if not all(np.isfinite(v) for v in metrics.values()):
        raise RuntimeError(f"Non-finite PPO metrics: {metrics}")
    if not all(torch.isfinite(p).all() for p in agent.policy.parameters()):
        raise RuntimeError("Non-finite policy parameters")
    last_update_metrics = dict(seconds=elapsed, metrics=metrics)
    update_times.append(elapsed)


agent._update = measured_update


def save_checkpoint(update):
    if frozen_state_normalization is not None:
        current = agent.checkpoint_modules['state_preprocessor'].state_dict()
        assert all(torch.equal(current[k], v) for k, v in frozen_state_normalization.items()), 'BC state normalization changed during PPO'
        progress['frozen_state_normalization_verified'] = True
    path = checkpoint_dir / f"update_{update:04d}.pt"
    temp = path.with_suffix(".tmp")
    agent.save(str(temp))
    from garmi_rl.joint_delta import is_joint_delta
    if is_joint_delta(agent_cfg):
        payload = torch.load(temp, map_location=env.device, weights_only=False)
        payload['action_contract'] = dict(name=agent_cfg['garmi_task']['action_space'],
            loss=agent_cfg.get('expert_bc',{}).get('loss','action_mse'))
        torch.save(payload,temp)
    temp.replace(path)
    state = dict(update=update, num_envs=args.num_envs, rollout=rollout,
                 checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 learning_config={"models": agent_cfg["models"], "agent": {
                     k: v for k, v in agent_cfg["agent"].items() if k != "experiment"}},
                 scheduler=agent.scheduler.state_dict(), scaler=agent.scaler.state_dict(),
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                 numpy_rng=np.random.get_state(), python_rng=random.getstate(), progress=progress)
    if "garmi_task" in agent_cfg:
        state["learning_config"]["garmi_task"] = agent_cfg["garmi_task"]
    if "expert_bc" in agent_cfg:
        state["learning_config"]["expert_bc"] = agent_cfg["expert_bc"]
        state['expert_index_sha256']=agent.expert_metadata['index_sha256']
    state_path = path.with_suffix(".resume.pt")
    temp_state = state_path.with_suffix(".tmp")
    torch.save(state, temp_state)
    temp_state.replace(state_path)
    progress["latest_checkpoint"] = str(path)
    atomic_json(out / "latest.json", {"update": update, "checkpoint": str(path),
                                     "resume": str(state_path), "sha256": state["checkpoint_sha256"]})
    return path


from functools import partial
from garmi_rl.trial_evaluation import (
    evaluate_checkpoint as run_fixed_evaluation,
    evaluate_paired_checkpoint as run_paired_evaluation,
)
evaluate_checkpoint = partial(run_fixed_evaluation, out=out, args=args, agent_cfg=agent_cfg, progress=progress)
evaluate_paired_checkpoint = partial(run_paired_evaluation, out=out, args=args, agent_cfg=agent_cfg, progress=progress)


try:
    print("TRIAL_READY", json.dumps({k: progress[k] for k in (
        "num_envs", "target_updates", "rollout", "seed", "eval_envs", "eval_seed")}), flush=True)
    segment_start = time.perf_counter()
    rollout_start = segment_start
    for timestep in range(resume_update * rollout, total_steps):
        agent.pre_interaction(timestep=timestep, timesteps=total_steps)
        with torch.no_grad():
            actions, _, action_outputs = agent.act(obs, timestep=timestep, timesteps=total_steps)
            if not fixed_finish:
                grip_mean = action_outputs["mean_actions"][:, -1]
                grip_std = agent.policy.distribution(role="policy").stddev[:, -1]
                close_probability = .5 * torch.erfc(grip_mean / (grip_std * (2.**.5)))
            torch.cuda.synchronize()
            tick = time.perf_counter()
            next_obs, reward, done, truncated, infos = wrapped.step(actions)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-tick
            if not torch.isfinite(next_obs).all() or not torch.isfinite(reward).all():
                raise RuntimeError("Non-finite observation/reward")
            agent.record_transition(states=obs, actions=actions, rewards=reward,
                next_states=next_obs, terminated=done, truncated=truncated, infos=infos,
                timestep=timestep, timesteps=total_steps)
            for key, value in infos.get("log", {}).items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    agent.track_data(f"Info / {key}", value.item())
        obs = next_obs
        d = env.diagnostics
        ended = (done | truncated).flatten()
        episode_return += reward.flatten()
        episode_length += 1
        terminal = torch.full((args.num_envs,), len(terminal_names), device=env.device, dtype=torch.long)
        # Higher-priority failure overrides success if both occur on the same step.
        for code, key in reversed(list(enumerate(terminal_names))):
            terminal[d[key]] = code
        ids = ended.nonzero(as_tuple=False).flatten()
        names = terminal_names + ("timeout",)
        step_counts = {name: int(((terminal == i) & ended).sum()) for i, name in enumerate(names)}
        counts.update(step_counts)
        if len(ids):
            columns=(ids, terminal[ids], episode_length[ids], episode_return[ids])
            if hasattr(env,"grasp_bank"):
                columns += (d["object_id"][ids],d["reference_id"][ids])
            records = torch.stack(columns, dim=1).cpu().tolist()
            for record in records:
                env_id, reason, length, value = record[:4]
                episode_log.write(dict(segment=segment_id, step=timestep+1, env=int(env_id),
                    reason=names[int(reason)], length=int(length), return_value=value,
                    **(dict(object=env.grasp_bank.objects[int(record[4])]["name"],
                        reference=env.grasp_bank.references[int(record[4])][int(record[5])]["id"])
                        if hasattr(env,"grasp_bank") else {})))
            episode_return[ids] = 0
            episode_length[ids] = 0
        controlled = (d["phase"] == 0) if fixed_finish else torch.ones_like(ended)
        # Scripted IK targets differ from policy commands by design, not clipping.
        measured = controlled & ~ended
        count_measured = measured.sum().clamp_min(1)
        arm_clipped = (env.action_manager.get_term("arm").processed_actions
            -(actions[:, 3:10]*1.5+env.scene["robot"].data.default_joint_pos[:, :7])).abs().gt(1e-5)
        row = dict(segment=segment_id, step=timestep+1, seconds=elapsed, num_envs=args.num_envs,
            warmup=timestep < resume_update*rollout+24,
            **{f"{k}_count": v for k, v in step_counts.items()}, reward=float(reward.mean()),
            reward_terms=json.dumps(env.reward_manager._step_reward.mean(0).cpu().tolist()),
            distance_mean=float(d["distance"].mean()), distance_min=float(d["distance"].min()),
            contact_count=int(d["contact"].sum()), max_lift=float(d["height"].max()),
            aligned_count=int(((d["distance"] < .015) & (d["angle"] < .15)).sum()),
            gripper_width_mean=float(d["gripper_width"].mean()),
            base_clip_fraction=float(((actions[:, :3].abs() > 1) & measured[:, None]).sum()/(count_measured*3)),
            arm_target_clip_fraction=float((arm_clipped & measured[:, None]).sum()/(count_measured*7)),
            base_reference_distance_mean=float((d["base_position"][:, :2]-env.scene.env_origins[:, :2]).norm(dim=-1).mean()))
        if fixed_finish:
            row.update(gripper_close_fraction=float((d["phase"] >= 2).float().mean()),
                phase_counts=json.dumps(torch.bincount(d["phase"], minlength=5).cpu().tolist()),
                handover_alignment_count=int(((d["phase"] == 1) & (d["phase_steps"] == 0)
                    & (d["handover_trigger"] == 1)).sum()),
                handover_deadline_count=int(((d["phase"] == 1) & (d["phase_steps"] == 0)
                    & (d["handover_trigger"] == 2)).sum()))
        else:
            row.update(gripper_close_fraction=float((actions[:, -1] < 0).float().mean()),
                gripper_mean_min=float(grip_mean.min()), gripper_mean_max=float(grip_mean.max()),
                gripper_std=float(grip_std.mean()), close_probability_mean=float(close_probability.mean()))
        step_log.write(row)
        step_times.append(elapsed)
        agent.post_interaction(timestep=timestep, timesteps=total_steps)
        if (timestep+1) % rollout:
            continue
        update = (timestep+1)//rollout
        update_log.write(dict(segment=segment_id, update=update, step=timestep+1,
            seconds=last_update_metrics["seconds"], rollout_and_update_seconds=time.perf_counter()-rollout_start,
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            metrics=json.dumps(last_update_metrics["metrics"])))
        transitions = resume_transitions + (timestep+1-resume_update*rollout)*args.num_envs
        progress.update(update=update, step=timestep+1, transitions=transitions,
            episode_counts=dict(counts), last_step=row, last_update=last_update_metrics,
            segment_elapsed_seconds=time.perf_counter()-segment_start, updated_unix=time.time())
        if update == resume_update+1 or update % 100 == 0:
            scaler = agent.checkpoint_modules.get("state_preprocessor")
            if scaler is not None and hasattr(scaler, "current_count"):
                z = (obs-scaler.running_mean.float())/(scaler.running_variance.float().sqrt()+scaler.epsilon)
                geometry_columns = list(range(24,37)) + (list(range(55,62)) if fixed_finish else [])
                progress["normalization_diagnostic"] = dict(update=update,
                    sample="current_training_batch", count=float(scaler.current_count),
                    any_feature_clipped_fraction=float((z.abs()>scaler.clip_threshold).any(1).float().mean()),
                    any_geometry_feature_clipped_fraction=float((z[:,geometry_columns].abs()>scaler.clip_threshold).any(1).float().mean()))
        if update % 10 == 0 or update == resume_update+1:
            print("TRIAL_PROGRESS", json.dumps({k: progress[k] for k in (
                "update", "transitions", "episode_counts", "segment_elapsed_seconds")}), flush=True)
        checkpoint = None
        budget_expired = args.wall_hours>0 and time.perf_counter()-segment_start>=args.wall_hours*3600
        scheduled_evaluation = (update in evaluation_updates if evaluation_updates is not None
                                else update % args.eval_interval == 0) or update == args.updates
        if scheduled_evaluation or stop_requested or budget_expired:
            checkpoint = save_checkpoint(update)
        if (scheduled_evaluation or budget_expired) and not stop_requested and not args.skip_evaluation:
            evaluate_checkpoint(checkpoint, update)
            if args.paired_eval_cases:
                evaluate_paired_checkpoint(checkpoint, update)
            # Persist evaluation history and RNG metadata alongside the same weights.
            save_checkpoint(update)
        atomic_json(out / "progress.json", progress)
        rollout_start = time.perf_counter()
        if budget_expired:
            progress["completion_reason"]="wall_clock_budget"
        if stop_requested or budget_expired:
            break
    progress["status"] = "interrupted" if stop_requested else "complete"
    segment.update(end_update=progress["update"], end_unix=time.time(),
        end_transitions=progress["transitions"],
        mean_step_ms=float(np.mean(step_times[24:] or step_times)*1000),
        mean_update_ms=float(np.mean(update_times)*1000))
    atomic_json(out / "progress.json", progress)
    atomic_json(out / "summary.json", progress)
    print("TRIAL_DONE", json.dumps({"status": progress["status"], "update": progress["update"],
        "best_evaluation": progress.get("best_evaluation")}), flush=True)
except BaseException as exc:
    progress.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    atomic_json(out / "progress.json", progress)
    traceback.print_exc()
    raise
finally:
    for log in (step_log, update_log, episode_log):
        log.close()
    if hasattr(agent, "writer"):
        agent.writer.flush()
        agent.writer.close()
    env.close()
    app.close()
