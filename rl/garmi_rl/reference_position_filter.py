"""Lookup a validated spatial reference mask once at episode reset."""
import hashlib
import json
from pathlib import Path

import torch


class ReferencePositionFilter:
    def __init__(self, path, bank_path, workspace_root, device='cpu', require_validated=True):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        if self.data['schema_version'] != 1:
            raise ValueError('Unsupported reference mask schema')
        if require_validated and self.data['status'] != 'validated':
            raise ValueError('Reference mask has not passed validation')
        root = Path(workspace_root)
        for relative, expected in self.data['sources_sha256'].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Reference mask source changed: {relative}')
        if hashlib.sha256(Path(bank_path).read_bytes()).hexdigest() != self.data['sources_sha256'][self.data['bank_path']]:
            raise ValueError('Reference bank differs from the spatial mask')
        bank = json.loads(Path(bank_path).read_text())
        obj = next(x for x in bank['objects'] if x['name'] == self.data['object_name'])
        if [r['id'] for r in obj['references']] != self.data['reference_ids']:
            raise ValueError('Reference mask indices do not match the reference bank')
        self.device = torch.device(device)
        self.mask = torch.zeros((len(self.data['cells']), len(self.data['reference_ids'])), dtype=torch.bool, device=self.device)
        for i, cell in enumerate(self.data['cells']):
            self.mask[i, cell['allowed_reference_indices']] = True
        if not self.mask.any(dim=1).all():
            raise ValueError('Reference mask contains an empty placement cell')

    def cell_indices(self, map_xy, map_yaw_degrees):
        xy = torch.as_tensor(map_xy, dtype=torch.float64, device=self.device)
        if xy.ndim != 2 or xy.shape[1] != 2 or not torch.isfinite(xy).all():
            raise ValueError('Expected finite map XY positions with shape (N, 2)')
        yaw = torch.as_tensor(map_yaw_degrees, dtype=torch.float64, device=self.device)
        try:
            yaw = torch.broadcast_to(yaw, (len(xy),))
        except RuntimeError as exc:
            raise ValueError('Yaw must be scalar or length N') from exc
        error = (yaw - self.data['bowl_map_yaw_degrees'] + 180.) % 360. - 180.
        if not torch.isfinite(yaw).all() or (error.abs() > 1.e-4).any():
            raise ValueError('This reference mask only supports bowl map yaw = 0 degrees')
        found = torch.full((len(xy),), -1, dtype=torch.long, device=self.device)
        tolerance = self.data['numeric_position_tolerance_m']
        for region in self.data['regions']:
            lower, upper = torch.tensor(region['bounds_map_xy'], dtype=xy.dtype, device=self.device)
            inside = ((xy >= lower - tolerance) & (xy <= upper + tolerance)).all(dim=1)
            shape = torch.tensor(region['shape_xy'], dtype=torch.long, device=self.device)
            local = torch.floor((xy.clamp(lower, upper) - lower) / self.data['cell_size_m']).long()
            local = torch.minimum(local, shape - 1)
            index = region['cell_offset'] + local[:, 0] * shape[1] + local[:, 1]
            found = torch.where(inside, index, found)
        if (found < 0).any():
            raise ValueError('Bowl position outside the validated placement regions')
        return found

    def allowed(self, map_xy, map_yaw_degrees, reference_pool=None):
        allowed = self.mask[self.cell_indices(map_xy, map_yaw_degrees)].clone()
        if reference_pool is not None:
            pool = torch.as_tensor(reference_pool, dtype=torch.long, device=self.device)
            if pool.ndim != 1 or not len(pool) or (pool < 0).any() or (pool >= self.mask.shape[1]).any():
                raise ValueError('Invalid reference pool')
            selected = torch.zeros(self.mask.shape[1], dtype=torch.bool, device=self.device)
            selected[pool] = True
            allowed &= selected
        if not allowed.any(dim=1).all():
            raise ValueError('No allowed reference at this position within the requested pool')
        return allowed

    def sample(self, map_xy, map_yaw_degrees, reference_pool=None, uniform=None):
        allowed = self.allowed(map_xy, map_yaw_degrees, reference_pool)
        count = allowed.sum(dim=1)
        if uniform is None:
            uniform = torch.rand(len(allowed), device=self.device)
        else:
            uniform = torch.as_tensor(uniform, dtype=torch.float64, device=self.device)
        if uniform.shape != count.shape or not torch.isfinite(uniform).all() or ((uniform < 0) | (uniform >= 1)).any():
            raise ValueError('Sampling uniforms must have shape (N,) and lie in [0, 1)')
        rank = (uniform * count).long()
        chosen = ((allowed.long().cumsum(dim=1) > rank[:, None]) & allowed).long().argmax(dim=1)
        return chosen
