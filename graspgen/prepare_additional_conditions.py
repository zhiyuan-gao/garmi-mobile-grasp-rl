"""Test previously untried successful placement/base conditions, not retries.

Failed repeatability conditions are excluded. Already retained references are
also excluded so computation targets new usable reference poses.
"""
import argparse
import copy
import json
from pathlib import Path
from select_references import ROOT,sha
from prepare_rl_validation import save


def main(batch,name):
    source=batch/name
    primary=json.loads((batch/f'repeat-{name}'/name/'review.json').read_text())
    tested={c['id'] for c in primary['conditions']}
    registry=json.loads((ROOT/'runs/validated-reference-bank-v4'/name/'all-validated-references.json').read_text())
    already={r['id'] for r in registry}
    original=batch/f'evidence-{name}'/name/'success-evidence.json'
    evidence=json.loads(original.read_text());selected=[]
    for row in evidence['references']:
        if row['reference'] in already:continue
        row=copy.deepcopy(row)
        row['successful_conditions']=[c for c in row['successful_conditions'] if
            f"{row['reference']}-{c['region']}-{c['depth']}" not in tested]
        if row['successful_conditions']:selected.append(row)
    assert selected, 'No untried successful conditions remain'
    evidence.update(references=selected,reference_count=len(selected),
        selection_scope='Only previously untried successful placement/base conditions of references not already retained; no repeated attempt at a failed repeatability condition.',
        complete_initial_evidence_sha256=sha(original))
    evidence_out=batch/f'extra-evidence-{name}'
    save(evidence_out/name/'success-evidence.json',evidence)
    import prepare_grasp_repeatability as module
    module.SOURCES={name:source};module.OUT=batch/f'repeat-{name}-extra';module.EVIDENCE=evidence_out
    module.main()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--batch',type=Path,required=True)
    p.add_argument('--object',choices=['mug','pitcher'],default='pitcher');a=p.parse_args()
    main(a.batch.resolve(),a.object)
