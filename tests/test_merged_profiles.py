"""Exercise the two asset formats and preserve historical action semantics."""
import hashlib
import json
from types import SimpleNamespace as S

import pytest
from garmi_rl.base_velocity import velocity_at_link, planning_root_state
from garmi_rl.expert_assets import load_expert_assets


def write_json(path, value):
    path.write_text(json.dumps(value))


def make_assets(root, name, separate):
    target = {'name': name, 'references': [{'id': name + '-r0'}]}
    for key in ['mesh', 'usd'] + (['cooked'] if separate else []):
        p = root / key
        p.write_bytes((name + key).encode())
        target[key] = str(p)
        target[key + '_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    bank = root / 'bank.json'
    write_json(bank, {'objects': [target]})
    config = None
    if not separate:
        cooked = root / 'full.pkl'
        cooked.write_bytes(b'full robot and object collision export')
        config = root / 'expert.json'
        write_json(config, {'object_name': name, 'bank': str(bank), 'cooked_model': 'full.pkl',
            'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [bank, cooked]}})
    return bank, config


@pytest.mark.parametrize('name,separate', [('cup', False), ('milk', False), ('mug', True)])
def test_loading_preserves_object_identity_and_geometry_format(tmp_path, name, separate):
    bank, config = make_assets(tmp_path, name, separate)
    assets = load_expert_assets(tmp_path, manifest=bank, object_name=name, config_path=config)
    assert assets['object_name'] == name
    assert list(assets['references']) == [name + '-r0']
    assert (assets['object_cooked'] is not None) == separate
    if config:
        assert load_expert_assets(tmp_path, config_path=config)['object_name'] == name
        assert assets['cooked_model'] == tmp_path / 'full.pkl'


def test_mismatched_object_or_bank_cannot_reuse_legacy_hulls(tmp_path):
    bank, config = make_assets(tmp_path, 'milk', False)
    with pytest.raises(ValueError, match='requested object'):
        load_expert_assets(tmp_path, manifest=bank, object_name='mug', config_path=config)
    other = tmp_path / 'other.json'
    other.write_bytes(bank.read_bytes())
    with pytest.raises(ValueError, match='reference bank'):
        load_expert_assets(tmp_path, manifest=other, object_name='milk', config_path=config)


@pytest.mark.parametrize('separate', [False, True])
def test_changed_geometry_is_rejected(tmp_path, separate):
    bank, config = make_assets(tmp_path, 'mug' if separate else 'milk', separate)
    (tmp_path / ('cooked' if separate else 'full.pkl')).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        load_expert_assets(tmp_path, manifest=bank, config_path=config,
                           object_name='mug' if separate else 'milk')


def test_non_bowl_cannot_silently_use_bowl_collision_export(tmp_path):
    bank, _ = make_assets(tmp_path, 'milk', False)
    with pytest.raises(ValueError, match='cooked geometry'):
        load_expert_assets(tmp_path, manifest=bank, object_name='milk')


def test_base_frame_is_explicit_and_state_matches_action():
    assert not velocity_at_link({})
    assert velocity_at_link({'base_velocity_frame': 'root_link'})
    with pytest.raises(ValueError):
        velocity_at_link({'base_velocity_frame': 'guess'})
    data = S(root_state_w='legacy', root_link_state_w='link')
    for value, expected in [(False, 'legacy'), (True, 'link')]:
        env = S(cfg=S(actions=S(base=S(velocity_at_link=value))), scene={'robot': S(data=data)})
        assert planning_root_state(env) == expected
