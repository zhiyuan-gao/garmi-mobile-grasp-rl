"""Summarize full-approach evidence without changing reference banks or quotas."""
import argparse
import collections
import csv
import json
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/ycb-expert-collection-v1'))
    args = p.parse_args()
    report = dict(objects={}, optimizer_updates=0,
                  scope='Full approach audits; audit successes are retained separately from quota datasets')
    for name in ('mug', 'pitcher'):
        out = args.root/name
        profile = yaml.safe_load((out/'task.yaml').read_text())
        refs = profile['garmi_task']['multi_object']['reference_ids'][name]
        rows = {ref: dict(reference=ref) for ref in refs}
        info = {}
        for stage, label in [('pilot', 'pilot'), ('expert-check-400', 'audit')]:
            path = out/stage/'summary.json'
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            successes = [r for r in data['results'] if r['reason'] == 'success']
            counts = collections.Counter(r['reference'] for r in successes)
            info[label] = dict(status=data['status'], cases_completed=len(data['results']),
                               counts=data['counts'], successful_references=len(counts),
                               saved_successful_trajectories=len(successes))
            for ref in refs:
                subset = [r for r in data['results'] if r['reference'] == ref]
                rows[ref][label+'_attempts'] = len(subset)
                for reason in ['success', 'collision', 'dropped', 'attempt_failed', 'timeout', 'expert_query_failed', 'out_of_bounds']:
                    rows[ref][label+'_'+reason] = sum(r['reason'] == reason for r in subset)
            if label == 'audit':
                info['missing_full_approach_success'] = sorted(set(refs)-set(counts))
                info['all_40_references_gate'] = ('passed' if data['status'] == 'complete' and len(counts) == 40
                                                  else 'failed' if data['status'] == 'complete' else 'pending')
                failures = [r for r in data['results'] if r['reason'] == 'attempt_failed']
                info['attempt_failure_diagnostics'] = dict(
                    lifted_with_contact_but_slip_limit_failed=sum(r['height'] >= .08 and r['contact'] and r['slip'] >= .02 for r in failures),
                    approach_timed_out=sum(r['handover_trigger'] == 2 for r in failures))
        for filename in ['coverage-gate.json', 'preflight-data-audit.json', 'collection-status.json']:
            if (out/filename).exists():
                info[filename.removesuffix('.json')] = json.loads((out/filename).read_text())
        with (out/'reference-approach-results.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values()))))
            writer.writeheader()
            writer.writerows(rows.values())
        report['objects'][name] = info
    path = args.root/'approach-review.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
