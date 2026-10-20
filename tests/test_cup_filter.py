"""Cup integration: correct bank, collision mask and evaluation isolation."""
import collections
import json
from pathlib import Path

import pytest
import torch

from garmi_rl.reference_position_filter import ReferencePositionFilter

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cup_filter():
    if not (ROOT/'assets/cup/position-mask.json').exists():
        pytest.skip('Prepare the cup inputs first')
    return ReferencePositionFilter(ROOT/'assets/cup/position-mask.json',
                                   ROOT/'assets/cup/bank.json', ROOT)


def test_all_cell_centres_match_certified_references(cup_filter):
    xy = [torch.tensor(c['bounds_map_xy']).double().mean(0).tolist()
          for c in cup_filter.data['cells']]
    allowed = cup_filter.allowed(xy, 0.)
    assert allowed.shape == (1040, 4)
    assert torch.equal(allowed, cup_filter.mask)
    assert (allowed.sum(1) >= 2).all()


def test_cup_mask_does_not_accept_bowl_bank():
    if not (ROOT/'assets/cup/position-mask.json').exists():
        pytest.skip('Prepare the cup inputs first')
    with pytest.raises(ValueError, match='Reference bank differs'):
        ReferencePositionFilter(ROOT/'assets/cup/position-mask.json',
                                ROOT/'assets/grasp_bank/selected-40.json', ROOT)


def test_outside_region_and_wrong_yaw_rejected(cup_filter):
    with pytest.raises(ValueError, match='outside'):
        cup_filter.allowed([[1.169, 7.43]], 0.)
    with pytest.raises(ValueError, match='fixed object map yaw'):
        cup_filter.allowed([[1.2, 7.43]], 10.)


def test_eval_and_audit_cases_balanced_allowed_and_disjoint(cup_filter):
    ids = cup_filter.data['reference_ids']
    evaluation = json.loads((ROOT/'runs/cup4-preppo/evaluation-cases.json').read_text())
    audit = json.loads((ROOT/'runs/cup4-preppo/expert-audit-cases.json').read_text())
    keys = set()
    for cases, per_reference in [(evaluation, 100), (audit, 20)]:
        assert collections.Counter(c['reference'] for c in cases) == dict.fromkeys(ids, per_reference)
        allowed = cup_filter.allowed([[.3-c['object_state'][1], 6.35+c['object_state'][0]]
                                      for c in cases], 0.)
        for i, case in enumerate(cases):
            assert case['object'] == 'cup'
            assert allowed[i, ids.index(case['reference'])]
            key = tuple(case['robot_state'] + case['object_state'])
            assert key not in keys
            keys.add(key)
