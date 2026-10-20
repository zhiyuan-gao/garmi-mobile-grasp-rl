"""Contact permission is limited to wrist/target, including contact history."""
import pytest
import torch
from garmi_rl.contact_policy import permitted_forces


def test_only_wrist_object_is_allowed_and_inputs_stay_unchanged():
    # Three environments: wrist/target only; also wrist/furniture; also other link contact.
    matrix = torch.zeros(3, 2, 1, 3, 3)
    matrix[:, 0, 0, 1, 0] = 12.  # target filter intentionally in the middle; historic force
    matrix[1, 1, 0, 0, 0] = 2.   # furniture
    forces = torch.tensor([[0., 12., 0.], [0., 12., 0.], [0., 12., 3.]])
    filtered, target = permitted_forces(forces, matrix, 1, 1, 'fingers_and_wrist')
    assert torch.equal(filtered, torch.tensor([[0., 0., 0.], [0., 2., 0.], [0., 0., 3.]]))
    assert torch.equal(target, torch.full((3,), 12.))
    assert torch.equal(forces[:, 1], torch.full((3,), 12.))
    strict, _ = permitted_forces(forces, matrix, 1, 1, 'fingers_only')
    assert torch.equal(strict, forces)


def test_wrist_self_collision_is_not_allowed():
    matrix = torch.zeros(1, 2, 1, 3, 3)
    matrix[0, 1, 0, 2, 1] = 4.
    result, _ = permitted_forces(torch.tensor([[4.]]), matrix, 0, 0, 'fingers_and_wrist')
    assert result.item() == 4.


def test_unknown_policy_fails_closed():
    with pytest.raises(ValueError, match='Unknown contact policy'):
        permitted_forces(torch.zeros(1, 1), torch.zeros(1, 1, 1, 2, 3), 0, 0, 'allow_everything')
