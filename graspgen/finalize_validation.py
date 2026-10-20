"""Publish the tested candidate version and a per-reference diagnostic report."""
import collections
import csv
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ycb-rl-validation'


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    report=dict(status='pre_expert_diagnostics_complete_awaiting_user_confirmation',
        expert_trajectories_generated=0,optimizer_updates=0,
        scope='All 80 references assessed for kitchen geometry. References with allowed positions tested by direct IK initialization and the original RL close/lift/hold controller. Giskard approach trajectories not executed.',
        limitations=['At least one successful diagnostic does not mean success across the kitchen or a reachable collision-free approach path.',
                    'Only finite IK seeds and four representative placements were tested; failures are retained.'],objects={})
    rows=[];objects=[]
    archive=OUT/'previous-published';archive.mkdir(exist_ok=True)
    for ycb,name in [('019_pitcher_base','pitcher'),('025_mug','mug')]:
        folder=OUT/ycb
        refs=json.loads((folder/'selected-40.json').read_text())
        physics=json.loads((folder/'physics/summary.json').read_text())
        mask=json.loads((folder/'position-mask.json').read_text())
        bank=json.loads((folder/'bank.json').read_text())
        handles=[]
        if name=='mug':
            selection=json.loads((folder/'selection-report.json').read_text())
            handles=[r['reference'] for r in selection['references'] if r['handle']]
            assert len(handles)==10
        assert len(refs)==40 and [r['id'] for r in refs]==mask['reference_ids']
        assert physics['configuration_sha256']==hashlib.sha256((folder/'task.yaml').read_bytes()).hexdigest()
        for j,ref in enumerate(refs):
            tests=physics['per_reference'][ref['id']]
            rows.append(dict(object=name,reference=ref['id'],handle=(ref['id'] in handles) if name=='mug' else None,
                allowed_cells=mask['allowed_cells_per_reference'][j],
                diagnostic_attempts=tests['attempts'],successful_attempts=tests['successes'],
                status='diagnostic_success' if tests['successes'] else 'no_allowed_position' if not mask['allowed_cells_per_reference'][j] else 'no_diagnostic_success',
                reasons=json.dumps(tests['reasons'],sort_keys=True)))
        report['objects'][name]=dict(total_references=40,
            references_with_allowed_positions=sum(c>0 for c in mask['allowed_cells_per_reference']),
            diagnostic_cases=physics['case_count'],references_with_diagnostic_success=physics['references_with_success'],
            case_results=physics['counts'],handle_references=handles if name=='mug' else None,
            handle_references_with_diagnostic_success=(sum(physics['per_reference'][r]['successes']>0 for r in handles) if name=='mug' else None),
            reference_sha256=hashlib.sha256((folder/'selected-40.json').read_bytes()).hexdigest(),
            physics_report=str((folder/'physics/summary.json').relative_to(ROOT)),
            mask_report=str((folder/'position-mask.json').relative_to(ROOT)),
            gallery=str((folder/'gallery-40.png').relative_to(ROOT)))
        target=ROOT/'resources'/f'{name}-40-references.json'
        if not (archive/target.name).exists():shutil.copy2(target,archive/target.name)
        shutil.copy2(folder/'selected-40.json',target)
        obj=bank['objects'][0]
        obj['validation']=report['objects'][name]
        objects.append(obj)
    audit=OUT/'collision-audit/review.json'
    if audit.exists():
        report['status']='diagnostics_complete_collision_fidelity_review_needed_expert_pending_confirmation'
        report['collision_geometry_review']=str(audit.relative_to(ROOT))
    save(OUT/'validation.json',report)
    selection_path=ROOT/'resources/ycb-40-reference-selection.json'
    if not (archive/selection_path.name).exists():shutil.copy2(selection_path,archive/selection_path.name)
    save(selection_path,report)
    bank_path=ROOT/'assets/grasp_bank/ycb-selected-40.json'
    if not (archive/'ycb-selected-40.json').exists():shutil.copy2(bank_path,archive/'ycb-selected-40.json')
    save(bank_path,dict(status=report['status'],objects=objects))
    with (OUT/'reference-validation.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(2,1,figsize=(17,6))
    for ax,name in zip(axes,['pitcher','mug']):
        rr=[r for r in rows if r['object']==name]
        colors=['#298e57' if r['successful_attempts'] else '#a84742' if r['allowed_cells']==0 else '#d5a331' for r in rr]
        ax.bar(np.arange(1,41),[r['successful_attempts'] for r in rr],color=colors)
        for i,r in enumerate(rr,1):
            ax.text(i,max(r['successful_attempts'],.1)+.08,f"{r['successful_attempts']}/{r['diagnostic_attempts']}",ha='center',fontsize=7)
            if not r['allowed_cells']:ax.scatter(i,.08,marker='x',c='#a84742')
        ax.set_xticks(range(1,41));ax.set_ylim(0,4.6);ax.set_ylabel('Successful trials');ax.set_xlabel('Reference ID suffix')
        ax.set_title(name+(' (01-10 = handle references)' if name=='mug' else ''))
    fig.suptitle('Direct-initialization RL diagnostics; no expert approach trajectories',fontsize=15)
    fig.tight_layout();fig.savefig(OUT/'validation-overview.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
