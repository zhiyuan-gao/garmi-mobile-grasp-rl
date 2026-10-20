"""Freeze an additional pitcher candidate batch before any new-policy results exist."""
from prepare_wrist_v6 import OUT
from prepare_wrist_approach import configure_cases
from prepare_rl_validation import save
from select_references import ROOT,sha
from try_handle_roll import read
from wrist_v6_inputs import build


def main():
    old={r['id'] for r in read(ROOT/'resources/validated-ycb-v5/pitcher/backup-validated-references.json')}
    best={}
    for batch in ['a','b','c']:
        folder=ROOT/f'runs/validated-reference-bank-v5/batch-{batch}'
        for c in read(folder/'pitcher/review.json')['conditions']:
            if c['reference'] in old:continue
            score=sum(c['successes'].values())
            if c['reference'] not in best or best[c['reference']][0]<score:
                best[c['reference']]=(score,folder,c)
    ranked=sorted(best.values(),key=lambda x:(-x[0],x[2]['reference']))
    selected=ranked[:32]
    # Two additional historical cases had >=2 s holds with wrist-only force violations.
    for ref in ['pitcher-v4e-n110','pitcher-v4c-c01543']:
        if ref not in {c['reference'] for _,_,c in selected}:selected.append(best[ref])
    rows=[]
    for score,folder,c in selected:
        source=folder/'source'
        ref=next(r for r in read(source/'bank.json')['objects'][0]['references'] if r['id']==c['reference'])
        rows.append(dict(ref=ref,source=str(source.relative_to(ROOT)),handle=c['handle'],conditions=[c],
            historical_review=str((folder/'pitcher/review.json').relative_to(ROOT)),
            historical_review_sha256=sha(folder/'pitcher/review.json'),historical_successes=score))
    assert not (OUT/'pitcher/extra').exists()
    folder=build('pitcher','extra',rows);configure_cases('pitcher',folder,'near_endpoint')
    save(folder/'selection.json',dict(method='32 previously unretained references ranked by historical physical successes, plus two wrist-only-force candidates; one best historical condition per reference',
        selection_frozen_before_new_policy_tests=True,old_success_labels_inherited=False,
        reference_count=len(rows),historical_failures_preserved=True))


if __name__=='__main__':main()
