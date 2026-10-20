"""Resolve legacy full-scene and YCB object-only expert collision exports."""
import hashlib
import json
from pathlib import Path


def load_expert_assets(root, *, manifest=None, object_name=None, config_path=None):
    root = Path(root)

    def path(value):
        value = Path(value)
        return value if value.is_absolute() else root / value

    def verify(filename, expected, label=None):
        if hashlib.sha256(filename.read_bytes()).hexdigest() != expected:
            if label is not None:
                raise ValueError(f'Expert object asset changed: {label}')
            raise ValueError(f'Expert input changed: {filename}')

    specification = json.loads(path(config_path).read_text()) if config_path else {}
    for name, digest in specification.get('files_sha256', {}).items():
        verify(path(name), digest)
    configured_name = specification.get('object_name')
    if object_name is not None and configured_name is not None and object_name != configured_name:
        raise ValueError('Expert configuration and requested object differ')
    object_name = object_name or configured_name or 'bowl'
    configured_bank = specification.get('bank')
    bank_path = path(manifest or configured_bank or 'assets/grasp_bank/selected-40.json')
    if manifest and configured_bank and bank_path.resolve() != path(configured_bank).resolve():
        raise ValueError('Expert configuration and requested reference bank differ')
    bank = json.loads(bank_path.read_text())
    matches = [o for o in bank['objects'] if o['name'] == object_name]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one expert object named {object_name}')
    target = matches[0]
    cooked_model = path(specification.get('cooked_model', 'assets/kitchen/cooked-model.pkl'))
    object_cooked = None
    if object_name != 'bowl' or manifest is not None:
        for key in ('mesh', 'usd'):
            verify(path(target[key]), target[key + '_sha256'], key)
        if 'cooked' in target:
            object_cooked = path(target['cooked'])
            verify(object_cooked, target['cooked_sha256'], 'cooked')
        elif 'cooked_model' not in specification:
            raise ValueError('Object bank requires cooked geometry or a verified full-scene expert config')
    if 'cooked_model' in specification:
        if specification['cooked_model'] not in specification.get('files_sha256', {}):
            raise ValueError('Expert cooked model must have a recorded hash')
    return dict(object_name=object_name, references={r['id']: r for r in target['references']},
                cooked_model=cooked_model, object_cooked=object_cooked, specification=specification)
