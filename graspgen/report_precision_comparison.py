"""Paired full-approach precision experiment; no training or bank replacement."""
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/ycb-expert-precision-v1'


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    return dict(min=min(values), median=float(np.median(values)), max=max(values)) if values else None


def main():
    plan = read(OUT/'plan.json')
    for filename, expected in plan['code_sha256'].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == expected, filename
    report = dict(status='complete', optimizer_updates=0, objects={},
                  limitation=plan['limitation'], default_configuration_changed=False)
    for obj in ('mug', 'pitcher'):
        out = OUT/obj
        data = {variant: read(out/variant/'summary.json') for variant in ('baseline', 'strict')}
        rows = {}
        stats = {}
        for variant, summary in data.items():
            assert summary['status'] == 'complete' and len(summary['results']) == 40
            assert summary['configuration_sha256'] == plan['objects'][obj]['configurations'][variant]
            rows[variant] = {r['reference']: r for r in summary['results']}
            assert len(rows[variant]) == 40
            aligned = [r for r in summary['results'] if r.get('handover_trigger') == 1]
            closed = [r for r in summary['results'] if r.get('final_phase', 0) >= 2]
            stats[variant] = dict(counts=summary['counts'],
                aligned_handover_cases=len(aligned),
                aligned_handover_distance_mm=distribution([r['handover_distance']*1000 for r in aligned]),
                aligned_handover_angle_rad=distribution([r['handover_angle'] for r in aligned]),
                close_start_distance_mm=distribution([r['close_start_distance']*1000 for r in closed]),
                close_start_angle_rad=distribution([r['close_start_angle'] for r in closed]),
                close_start_outside_original_endpoint_tolerance=sum(r['close_start_distance'] > .005 or r['close_start_angle'] > .05 for r in closed),
                closed_cases=len(closed))
            if variant == 'strict':
                assert summary['expert_precision'] == dict(translation_threshold=.002, orientation_threshold=.02)
                assert all(r['handover_distance'] < .005 and r['handover_angle'] < .05 for r in aligned)
                assert all(r['handover_trigger'] == 1 for r in summary['results'] if r['reason'] == 'success')
                assert all(r['final_phase'] == 0 and r['handover_trigger'] == 2 for r in summary['results'] if r['reason'] == 'approach_timeout')
                stats[variant]['strict_handover_and_timeout_assertions'] = 'passed'
        assert rows['baseline'].keys() == rows['strict'].keys()
        pairs = []
        for ref, baseline in rows['baseline'].items():
            strict = rows['strict'][ref]
            for key in ('robot_state', 'object_state'):
                assert baseline['initial_case'][key] == strict['initial_case'][key]
            pairs.append(dict(reference=ref, baseline=baseline['reason'], strict=strict['reason'],
                baseline_handover_mm=baseline.get('handover_distance', 0)*1000,
                strict_handover_mm=strict.get('handover_distance', 0)*1000,
                baseline_close_mm=baseline.get('close_start_distance', 0)*1000,
                strict_close_mm=strict.get('close_start_distance', 0)*1000))
        with (out/'paired-results.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
        changed = collections.Counter(('success' if p['baseline'] == 'success' else 'failure',
                                       'success' if p['strict'] == 'success' else 'failure') for p in pairs)
        stats['paired_transitions'] = {a+' -> '+b: n for (a, b), n in changed.items()}
        stats['improved_references'] = [p['reference'] for p in pairs if p['baseline'] != 'success' and p['strict'] == 'success']
        stats['regressed_references'] = [p['reference'] for p in pairs if p['baseline'] == 'success' and p['strict'] != 'success']
        for variant in data:
            path = out/f'{variant}-data-audit.json'
            if path.exists():
                stats[variant]['data_audit'] = read(path)
        report['objects'][obj] = stats
    (OUT/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
