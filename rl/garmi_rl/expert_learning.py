"""Offline expert data and BC loss, using the unchanged 64-state / 10-action interface."""
import collections
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from skrl.utils.runner.torch import Runner
from .expert_actions import COMMAND_VERSION
from .giskard_io import COMMAND_VERSION as GISKARD_COMMAND_VERSION
from .joint_delta import COMMAND_VERSION as DELTA_VERSION
from .augmentation import approach_label_mask


def check_expert_extension(previous_index, next_index):
    """A resumed DAgger round may append data, never silently replace old labels."""
    old=json.loads(Path(previous_index).read_text())
    new=json.loads(Path(next_index).read_text())
    def trajectory_keys(data):
        return {(r['id'],r['file'],r['sha256'],r.get('label_key','actions')) for r in data.get('demonstrations',[])}
    def label_keys(data):
        return {(r['file'],r['sha256']) for r in data.get('dagger_label_sets',[])}
    assert trajectory_keys(old)<=trajectory_keys(new),'Existing expert trajectories were changed or removed'
    assert label_keys(old)<=label_keys(new),'Existing DAgger labels were changed or removed'
    return dict(previous_index=str(previous_index),next_index=str(next_index),
        previous_sha256=hashlib.sha256(Path(previous_index).read_bytes()).hexdigest(),
        next_sha256=hashlib.sha256(Path(next_index).read_bytes()).hexdigest(),
        append_only_verified=True)


class ReferenceBalancedSampler:
    """Uniform reference, then uniform labelled timestep within that reference."""
    def __init__(self, references, device):
        self.names, inverse, counts = np.unique(references, return_inverse=True, return_counts=True)
        self.device = torch.device(device)
        self.order = torch.as_tensor(np.argsort(inverse, kind='stable'), device=self.device)
        self.counts = torch.as_tensor(counts, device=self.device)
        self.offsets = self.counts.cumsum(0)-self.counts

    def sample(self, count):
        ref = torch.randint(len(self.names), (count,), device=self.device)
        local = (torch.rand(count, device=self.device)*self.counts[ref]).long()
        return self.order[self.offsets[ref]+local]


def load_demonstrations(index, device, include_full_states=True):
    manifest = json.loads(Path(index).read_text())
    rows = manifest.get('demonstrations',[])
    label_sets = manifest.get('dagger_label_sets',[])
    dimension = 10
    for source in label_sets:
        if source.get('command_version') not in (COMMAND_VERSION, GISKARD_COMMAND_VERSION, DELTA_VERSION):
            raise ValueError('Obsolete DAgger action labels: regenerate with build_dagger_labels.py '
                             f'({COMMAND_VERSION}); do not mix restarted-cubic labels into a new run')
    if manifest.get('expert_backend') == 'giskard':
        expected = DELTA_VERSION if manifest.get('command_version') == DELTA_VERSION else GISKARD_COMMAND_VERSION
        assert all(r.get('command_version') == expected for r in rows+label_sets), 'Mixed expert action contracts'
    assert rows or label_sets
    assert len({r['id'] for r in rows}) == len(rows)
    observations, actions, full_states, references = [], [], [], []
    for row in rows:
        path = Path(row['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
        with np.load(path, allow_pickle=False) as data:
            x, y, mask = data['observations'], data[row.get('label_key','actions')], data['bc_mask']
            assert x.shape == (len(y) + 1, 64) and y.shape[1] == dimension
            noise = data['perturbation_mask'] if 'perturbation_mask' in data else None
            if row.get('augmentation') == 'ga_ddpg_dart':
                assert noise is not None, 'DART trajectories must distinguish perturbation actions'
            assert np.array_equal(mask, approach_label_mask(data['phases'], noise))
            assert np.isfinite(x).all() and np.isfinite(y).all() and mask.sum() == row['bc_steps']
            observations.append(x[:-1][mask].copy())
            actions.append(y[mask].copy())
            if include_full_states:
                full_states.append(x.copy())
            references.extend([row['reference']] * int(mask.sum()))
    label_ids=set()
    label_parents=set()
    for source in label_sets:
        path=Path(source['file'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256'],path
        samples=source['samples']
        with np.load(path,allow_pickle=False) as data:
            x,y=data['states'],data['expert_actions']
            assert x.shape==(len(samples),64) and y.shape==(len(samples),dimension)
            assert len(samples)==source['count'] and np.isfinite(x).all() and np.isfinite(y).all()
            assert np.all(x[:,49]==1.)
            observations.append(x.copy());actions.append(y.copy())
            if include_full_states:
                full_states.append(x.copy())
        for sample in samples:
            assert sample['id'] not in label_ids
            assert sample['expert_action_executed'] is False
            label_ids.add(sample['id'])
            case=sample['initial_case']
            label_parents.add(case.get('source_initial_id',case['id']))
            references.append(sample['reference'])
    return dict(states=torch.as_tensor(np.concatenate(observations), device=device),
                actions=torch.as_tensor(np.concatenate(actions), device=device),
                full_states=(torch.as_tensor(np.concatenate(full_states), device=device) if include_full_states else None),
                references=np.array(references), rows=rows,dagger_parent_ids=label_parents,
                metadata=dict(trajectories=len(rows),dagger_labels=len(label_ids), steps=len(references),
                    per_reference=dict(collections.Counter(r['reference'] for r in rows)),
                    label_steps_per_reference=dict(collections.Counter(references)),
                    index=str(index), index_sha256=hashlib.sha256(Path(index).read_bytes()).hexdigest()))


def policy_mean(policy, normalized_states):
    mean, _, _ = policy.compute({'states': normalized_states}, role='policy')
    # skrl shared models cache the last actor features for the subsequent value
    # call. An auxiliary expert batch must never be reused for an on-policy value.
    if hasattr(policy, '_shared_output'):
        policy._shared_output = None
    return mean


def cloning_loss(policy, preprocessor, states, actions, pose_loss=None):
    predicted = policy_mean(policy, preprocessor(states))
    if pose_loss is not None:
        return pose_loss(predicted, actions, states)
    return (predicted - actions).square().mean()


class ExpertRunner(Runner):
    def _component(self, name):
        if name.lower() == 'ppo' and self._cfg.get('expert_bc'):
            from .expert_ppo import ExpertPPO
            return ExpertPPO
        return super()._component(name)

    def _generate_agent(self, env, cfg, models):
        agent = super()._generate_agent(env, cfg, models)
        if cfg.get('expert_bc'):
            settings = copy.deepcopy(cfg['expert_bc'])
            from .joint_delta import is_joint_delta, JointPointLoss
            manifest = json.loads(Path(settings['index']).read_text())
            assert (manifest.get('command_version') == DELTA_VERSION) == is_joint_delta(cfg), 'Expert data action semantics mismatch'
            data_device = settings.get('data_device', env.device)
            data = load_demonstrations(settings['index'], data_device, include_full_states=False)
            agent.expert_states = data['states']
            agent.expert_actions = data['actions']
            agent.expert_settings = settings
            agent.expert_metadata = data['metadata']
            sampling = settings.get('sampling', 'uniform_steps')
            if sampling not in ('uniform_steps', 'uniform_references'):
                raise ValueError(f'Unsupported expert sampling: {sampling}')
            if sampling == 'uniform_references':
                agent.expert_sampler = ReferenceBalancedSampler(data['references'], data['states'].device)
            agent.expert_metadata.update(sampling=sampling, data_device=str(data['states'].device))
            if settings.get('loss', 'action_mse') == 'joint_point':
                assert is_joint_delta(cfg)
                agent.expert_pose_loss = JointPointLoss(env.device)
        return agent
