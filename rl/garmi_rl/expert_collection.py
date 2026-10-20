"""Strict demonstration acceptance and local, split-preserving quota sampling."""
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def strict_success(row):
    return (row.get('reason') == 'success' and row.get('handover_trigger') == 1
            and row.get('handover_distance', np.inf) <= .005 and row.get('handover_angle', np.inf) <= .05
            and row.get('close_start_distance', np.inf) <= .005 and row.get('close_start_angle', np.inf) <= .05
            and row.get('height', -np.inf) >= .08 and row.get('contact') is True and row.get('supported') is False
            and row.get('slip', np.inf) < .02 and row.get('hold_seconds', 0) >= 2
            and row.get('collision_bodies') == {})


def position_key(case):
    return np.array(case['robot_state'][:2] + case['object_state'][:2], dtype=float)


class LocalCaseSampler:
    """Draw near previously verified starts; assign split before simulation.

    All used initial states are separated in the four varying XY coordinates,
    even across references. The split never changes after observing an outcome.
    Extra concurrent attempts can finish a small outstanding quota efficiently;
    the collector still accepts only the first required successes.
    """
    def __init__(self, specification, seed, excluded=()):
        self.specification = specification
        self.rng = np.random.default_rng(seed)
        self.windows = specification['references']
        self.radius = specification['xy_radius_m']
        assert 0 < self.radius <= .004
        self.lo, self.hi = np.array(specification['base_bounds_task_xy'])
        self.separation = specification.get('minimum_state_separation_m', .0002)
        assert self.separation >= .0002
        self.oversampling = specification.get('parallel_attempts_per_needed_success', 4)
        assert 1 <= self.oversampling <= 8
        self.history, self.pending, self.tree = [], [], None
        self.generated = 0
        for case in excluded:
            self.add_used(case)

    def add_used(self, case):
        self.pending.append(position_key(case))
        if len(self.pending) >= 256:
            self.history.extend(self.pending)
            self.pending = []
            self.tree = cKDTree(np.array(self.history))

    def separated(self, point):
        if self.tree is not None and self.tree.query(point, p=np.inf)[0] <= self.separation:
            return False
        return not self.pending or np.max(np.abs(np.array(self.pending)-point), axis=1).min() > self.separation

    def draw(self, reference, split):
        window = self.windows[reference][int(self.rng.integers(len(self.windows[reference])))]
        for _ in range(10000):
            case = copy.deepcopy(window['case'])
            assert case['reference'] == reference
            case['robot_state'][:2] = np.clip(np.array(case['robot_state'][:2]) +
                self.rng.uniform(-self.radius, self.radius, 2), self.lo, self.hi).tolist()
            case['object_state'][:2] = (np.array(case['object_state'][:2]) +
                self.rng.uniform(-self.radius, self.radius, 2)).tolist()
            if self.separated(position_key(case)):
                self.add_used(case)
                self.generated += 1
                case.update(split=split, source_window=window['id'])
                return case
        raise RuntimeError('Local case sampler exhausted separated initial states')

    def sample(self, count, references, accepted, quotas):
        assigned = {(s, r): 0 for s in quotas for r in references}
        result = []
        for _ in range(count):
            pending = [(s, r) for s in quotas for r in references
                       if assigned[s, r] < max(0, quotas[s]-accepted[s].get(r, 0))*self.oversampling]
            if not pending:
                break
            score = lambda pair: (accepted[pair[0]].get(pair[1], 0) +
                                  assigned[pair]/self.oversampling) / quotas[pair[0]]
            smallest = min(map(score, pending))
            tied = [pair for pair in pending if abs(score(pair)-smallest) < 1.e-12]
            split, reference = tied[int(self.rng.integers(len(tied)))]
            result.append(self.draw(reference, split))
            assigned[split, reference] += 1
        return result


def excluded_cases(specification):
    cases = []
    for path in specification.get('excluded_case_files', []):
        cases.extend(json.loads(Path(path).read_text()))
    for path in specification.get('excluded_result_files', []):
        cases.extend(row['initial_case'] for row in json.loads(Path(path).read_text())['results'])
    return cases
