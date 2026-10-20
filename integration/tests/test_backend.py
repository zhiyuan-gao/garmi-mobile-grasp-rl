from types import SimpleNamespace

import pytest
import torch

from garmi_cram_rl.lab_backend import LabBackend
from sim import ensure_training_idle


def test_policy_uses_original_observation_and_raw_actions_then_reports_physical_success():
    backend = LabBackend.__new__(LabBackend)
    observation = torch.arange(64, dtype=torch.float32).reshape(1, 64)
    raw_action = torch.full((1, 10), 1.2)
    events = []
    backend.env = SimpleNamespace(device='cpu', episode_length_buf=torch.zeros(1), step_dt=.05,
        cfg=SimpleNamespace(episode_length_s=39.), grasp_state={'hold': torch.tensor([2.05])},
        action_manager=SimpleNamespace(process_action=lambda action: events.append(('action', action))))
    diagnostics = {key: torch.tensor([False]) for key in
                   ['collision', 'dropped', 'out_of_bounds', 'attempt_failed', 'supported']}
    diagnostics.update(success=torch.tensor([True]), height=torch.tensor([.12]),
                       slip=torch.tensor([.005]), contact=torch.tensor([True]))
    backend.mdp = SimpleNamespace(observation=lambda env: observation, diagnostics=lambda env: diagnostics)
    def policy(value):
        assert torch.equal(value, observation)
        return raw_action
    backend.policy = policy
    backend._physics = lambda policy: events.append(('physics', policy))
    assert backend.step_policy() == 'success'
    assert torch.equal(events[0][1], raw_action)  # Do not clip previous-action observations.
    assert events[1] == ('physics', True)
    assert backend.env.episode_length_buf.item() == 1
    assert backend.last_measurements['hold_seconds'] == pytest.approx(2.05)
    # A forbidden contact wins over a simultaneous success flag.
    diagnostics['collision'][:] = True
    assert backend.step_policy() == 'collision'


def test_physics_advances_ten_substeps_without_episode_autoreset():
    backend = LabBackend.__new__(LabBackend)
    events = []
    backend.env = SimpleNamespace(cfg=SimpleNamespace(decimation=10), physics_dt=.005,
        _sim_step_counter=0, common_step_counter=5,
        action_manager=SimpleNamespace(apply_action=lambda: events.append('act')),
        scene=SimpleNamespace(write_data_to_sim=lambda: events.append('write'),
                              update=lambda dt: events.append(('update', dt))),
        sim=SimpleNamespace(step=lambda render: events.append(('step', render))))
    backend.control_step, backend.render = 0, False
    backend._physics(policy=True)
    assert events == ['act', 'write', ('step', False), ('update', .005)] * 10
    assert backend.env.common_step_counter == 6
    assert backend.env._sim_step_counter == 10


def test_return_to_cram_keeps_closed_finger_targets_and_clears_cached_commands():
    backend = LabBackend.__new__(LabBackend)
    backend.targets = torch.tensor([[.7, 0., 0.]])
    backend.arm, backend.fingers = [0], [1, 2]
    backend.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[.8, .02, .02]])))
    backend.command_velocity = torch.ones_like(backend.targets)
    import numpy as np
    backend.base_command = np.ones(3)
    backend.return_to_cram()
    assert torch.equal(backend.targets, torch.tensor([[.8, 0., 0.]]))
    assert torch.count_nonzero(backend.command_velocity) == 0
    assert backend.base_command.tolist() == [0., 0., 0.]


def test_classical_arm_motion_does_not_reopen_closed_fingers():
    import numpy as np
    backend = LabBackend.__new__(LabBackend)
    backend.targets = torch.tensor([[.2, 0., 0.]])
    backend.command_velocity = torch.tensor([[.1, 0., 0.]])
    backend.base_command, backend.base_time, backend.command_time = np.zeros(3), 0., 0.
    backend.base_term = SimpleNamespace(_processed_actions=torch.zeros(1, 3), velocity=torch.zeros(1, 3))
    backend.env = SimpleNamespace(step_dt=.05)
    backend.robot = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.tensor([[.2, .02, .02]]),
        joint_pos_limits=torch.tensor([[[-3., 3.], [0., .04], [0., .04]]])))
    backend._physics = lambda policy: None
    backend.step_classical(.05)
    assert backend.targets[0, 0].item() == pytest.approx(.205)
    assert backend.targets[0, 1:].tolist() == [0., 0.]


def test_simulator_launch_is_blocked_while_training_runs(tmp_path):
    path = tmp_path / 'results/rl-trials/trial-17-bowl-kitchen-40refs-expert-ppo/supervisor.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"status": "running"}')
    with pytest.raises(RuntimeError, match='training is running'):
        ensure_training_idle(tmp_path)
    path.write_text('{"status": "complete"}')
    ensure_training_idle(tmp_path)
