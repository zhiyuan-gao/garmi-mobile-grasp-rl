"""Behavior cloning pretraining without starting Isaac Sim. Produces skrl checkpoints."""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from gymnasium.spaces import Box
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import shared_model
from garmi_rl.expert_learning import load_demonstrations, policy_mean
from garmi_rl.joint_delta import is_joint_delta, JointPointLoss, COMMAND_VERSION as DELTA_VERSION

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--index', required=True)
p.add_argument('--validation', required=True)
p.add_argument('--out', required=True)
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--batch-size', type=int, default=1024)
p.add_argument('--learning-rate', type=float, default=0.0003)
p.add_argument('--patience', type=int, default=20)
p.add_argument('--seed', type=int, default=20261028)
p.add_argument('--history-dropout', type=float, default=0.,
    help='Diagnostic BC regularization: probability of hiding previous commands during training only')
p.add_argument('--base-std',type=float,default=.1)
p.add_argument('--arm-std',type=float,default=.05)
p.add_argument('--checkpoint-epochs', type=int, nargs='*', default=[])
p.add_argument('--loss', choices=['action_mse', 'joint_point'], default='action_mse')
p.add_argument('--init-checkpoint', help='Continue actor and frozen normalization from an existing compatible checkpoint')
p.add_argument('--balance-references', action='store_true', help='Uniform reference then uniform labelled timestep per draw')
args = p.parse_args()
assert 0. <= args.history_dropout <= 1.
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.set_float32_matmul_precision('high')
device = 'cuda:0'
started = time.time()
profile = yaml.safe_load(Path(args.config).read_text())
training = load_demonstrations(args.index, device, include_full_states=not bool(args.init_checkpoint))
validation = load_demonstrations(args.validation, device, include_full_states=False)
assert not ({r['id'] for r in training['rows']} & {r['id'] for r in validation['rows']})
assert not (training['dagger_parent_ids'] & {r['initial_case'].get('source_initial_id',r['id']) for r in validation['rows']})
assert profile['models']['separate'] is False
models = copy.deepcopy(profile['models']); models.pop('separate')
roles = list(models)
structure = [models[role].pop('class') for role in roles]
action_dim = 10
joint_delta = is_joint_delta(profile)
point_loss = args.loss == 'joint_point'
if point_loss and not joint_delta:
    raise ValueError('Forward-kinematic point loss requires the joint-increment interface')
for source in (args.index, args.validation):
    version = json.loads(Path(source).read_text()).get('command_version')
    assert (version == DELTA_VERSION) == joint_delta, 'Dataset action semantics do not match the environment'
assert training['actions'].shape[1] == validation['actions'].shape[1] == action_dim
obs_space, action_space = Box(-np.inf, np.inf, (64,)), Box(-np.inf, np.inf, (action_dim,))


def make_policy():
    model = shared_model(observation_space=obs_space, action_space=action_space,
        device=device, structure=structure, roles=roles, parameters=[models[r] for r in roles])
    for role in roles:
        model.init_state_dict(role)
    return model


policy = make_policy()
state_scaler = RunningStandardScaler(64, device=device)
value_scaler = RunningStandardScaler(1, device=device)
initialization = None
if args.init_checkpoint:
    saved = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
    expected = profile['garmi_task'].get('action_space')
    assert saved.get('action_contract', {}).get('name') == expected, 'Warm-start action contract differs'
    policy.load_state_dict(saved['policy'])
    state_scaler.load_state_dict(saved['state_preprocessor'])
    value_scaler.load_state_dict(saved['value_preprocessor'])
    initialization = dict(checkpoint=args.init_checkpoint,
        sha256=hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
        normalization='inherited unchanged', optimizer='fresh Adam; BC checkpoints have no optimizer state')
with torch.no_grad():
    # All phases from TRAIN trajectories only. Fixed phases are never action labels.
    if not args.init_checkpoint:
        states = training['full_states'].double()
        state_scaler.running_mean.copy_(states.mean(0))
        state_scaler.running_variance.copy_(states.var(0, unbiased=False).clamp_min(1.e-4))
        state_scaler.current_count.fill_(len(states))
        del states
    # BC trains no critic. Reset its final layer; retain any inherited scaler.
    policy.value_layer.weight.zero_(); policy.value_layer.bias.zero_()
    policy.log_std_parameter.copy_(torch.log(torch.tensor([args.base_std] * 3 + [args.arm_std] * (action_dim-3), device=device)))
policy.log_std_parameter.requires_grad_(False)
for parameter in policy.value_layer.parameters():
    parameter.requires_grad_(False)
optimizer = torch.optim.Adam([v for v in policy.parameters() if v.requires_grad], lr=args.learning_rate)
train_x, train_y = state_scaler(training['states']), training['actions']
val_x, val_y = state_scaler(validation['states']), validation['actions']
train_raw, val_raw = training['states'], validation['states']
reference_sampler = None
if args.balance_references:
    from garmi_rl.expert_learning import ReferenceBalancedSampler
    reference_sampler = ReferenceBalancedSampler(training['references'], device)
pose_loss = JointPointLoss(device) if point_loss else None
zero_history = state_scaler(torch.zeros(1,64,device=device))[:,39:49]
del training['full_states'], validation['full_states']
history = []


def atomic(path, data):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2, allow_nan=False)); tmp.replace(path)


@torch.no_grad()
def metrics(x, y, references, raw):
    errors = torch.cat([policy_mean(policy, x[i:i + 4096]) - y[i:i + 4096]
                       for i in range(0, len(x), 4096)])
    squared = errors.square().mean(1)
    if point_loss:
        values = pose_loss.metrics(errors+y, y, raw)
        return dict(mse=float(squared.mean()), **{k: float(v) for k, v in values.items()},
            joint_delta_mae_rad=float(errors[:, 3:].abs().mean()*.05),
            base_command_mae=float(errors[:, :3].abs().mean()),
            per_reference={ref: float(pose_loss((errors+y)[references == ref], y[references == ref], raw[references == ref]))
                           for ref in sorted(set(references))})
    return dict(mse=float(squared.mean()),
        arm_target_mae_rad=float(errors[:, 3:].abs().mean() * (.05 if joint_delta else 1.5)),
        base_command_mae=float(errors[:, :3].abs().mean()),
        per_reference={ref: float(squared[torch.as_tensor(references == ref, device=device)].mean())
                       for ref in sorted(set(references))})


def checkpoint(path, epoch):
    saved = dict(policy=policy.state_dict(), value=policy.state_dict(),
        state_preprocessor=state_scaler.state_dict(), value_preprocessor=value_scaler.state_dict())
    if joint_delta:
        saved['action_contract'] = dict(name=profile['garmi_task']['action_space'], loss=args.loss)
    tmp = path.with_suffix('.tmp'); torch.save(saved, tmp); tmp.replace(path)
    # Load in a fresh model and compare deterministic output before accepting the checkpoint.
    restored = make_policy(); restored.load_state_dict(torch.load(path, weights_only=False)['policy'])
    with torch.no_grad():
        error = float((policy_mean(restored, val_x[:128]) - policy_mean(policy, val_x[:128])).abs().max())
    assert error == 0., error
    return dict(epoch=epoch, checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checkpoint_reload_max_error=error)


manifest = dict(status='training', pid=os.getpid(), args=vars(args), training=training['metadata'],
    initialization=initialization,
    validation=validation['metadata'], observation_dimension=64, action_dimension=action_dim,
    loss='hand point L1 / .01 m + base MSE + .1 * joint delta MSE' if point_loss else 'action MSE',
    action_space=profile['garmi_task'].get('action_space', 'base_velocity_absolute_joint_target'),
    action_labels='manifest label_key: executed expert actions or explicitly recorded corrective targets; approach only',
    normalization='inherited from initialization checkpoint; frozen' if args.init_checkpoint else 'fit full states of training trajectories only, variance floor 1e-4; frozen',
    value_initialization='zero final head; inherited value normalizer' if args.init_checkpoint else 'zero final head; fresh value normalizer',
    reference_sampling='uniform_references' if args.balance_references else 'uniform_steps',
    exploration_std=[args.base_std] * 3 + [args.arm_std] * (action_dim-3), history=history)
(out / 'config.yaml').write_text(yaml.safe_dump(profile))
initial = metrics(val_x, val_y, validation['references'], val_raw)
best_loss = float('inf'); best_epoch = 0
selection_key = 'loss' if point_loss else 'mse'
atomic(out / 'progress.json', {**manifest, 'initial_validation': initial})
with (out / 'epochs.jsonl').open('w', buffering=1) as log:
    for epoch in range(1, args.epochs + 1):
        tick = time.time(); total_loss = 0.; seen = 0
        permutation = (reference_sampler.sample(len(train_x)) if reference_sampler is not None
                       else torch.randperm(len(train_x), device=device))
        for ids in permutation.split(args.batch_size):
            batch_x = train_x[ids]
            if args.history_dropout:
                hidden = torch.rand(len(ids),device=device) < args.history_dropout
                batch_x[hidden,39:49] = zero_history
            predicted = policy_mean(policy, batch_x)
            loss = pose_loss(predicted, train_y[ids], train_raw[ids]) if point_loss else (predicted-train_y[ids]).square().mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids); seen += len(ids)
        validation_metrics = metrics(val_x, val_y, validation['references'], val_raw)
        row = dict(epoch=epoch, **{('train_loss' if point_loss else 'train_mse'):total_loss/seen}, validation=validation_metrics,
                   seconds=time.time()-tick, wall_seconds=time.time()-started)
        history.append(row); log.write(json.dumps(row) + '\n')
        if validation_metrics[selection_key] < best_loss - 1.e-7:
            best_loss, best_epoch = validation_metrics[selection_key], epoch
            manifest['best'] = checkpoint(out / 'best.pt', epoch)
            manifest['best']['validation'] = validation_metrics
        manifest.update(epoch=epoch, wall_seconds=time.time()-started, initial_validation=initial)
        atomic(out / 'progress.json', manifest)
        if epoch in args.checkpoint_epochs:
            checkpoint(out/f'epoch-{epoch:04d}.pt', epoch)
        print('BC_EPOCH', json.dumps(row), flush=True)
        if epoch >= 20 and epoch - best_epoch >= args.patience:
            manifest['stop_reason'] = 'validation_patience'
            break
manifest.update(status='complete', final=checkpoint(out / 'last.pt', epoch),
                best_validation_reduction=1.-best_loss/initial[selection_key])
atomic(out / 'progress.json', manifest); atomic(out / 'summary.json', manifest)
print('BC_DONE', json.dumps({k: manifest[k] for k in ('best', 'wall_seconds', 'best_validation_reduction')}), flush=True)
