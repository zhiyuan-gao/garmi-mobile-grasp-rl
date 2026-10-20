"""Prepare reviewed Giskard audit inputs only. Does not launch any trajectories."""
import hashlib
import json
import random
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ycb-rl-validation'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def inside(p):return '/workspace/'+str(p.relative_to(ROOT))


def main():
    for ycb,name in [('019_pitcher_base','pitcher'),('025_mug','mug')]:
        folder=OUT/ycb
        bank=json.loads((folder/'bank.json').read_text())
        mask=json.loads((folder/'position-mask.json').read_text())
        obj=bank['objects'][0]
        cooked=ROOT/'assets/ycb'/ycb/'cooked.pkl'
        meta=json.loads(cooked.with_suffix('.json').read_text())
        assert meta['usd_sha256']==obj['usd_sha256'] and meta['cooked_sha256']==sha(cooked)
        obj.update(cooked=inside(cooked),cooked_sha256=sha(cooked))
        bank['status']='expert_preflight_pending_user_confirmation'
        save(folder/'expert-bank.json',bank)
        mask['sources_sha256'].pop(mask['bank_path'])
        mask['bank_path']=str((folder/'expert-bank.json').relative_to(ROOT))
        mask['sources_sha256'][mask['bank_path']]=sha(folder/'expert-bank.json')
        mask['sources_sha256'][str(cooked.relative_to(ROOT))]=sha(cooked)
        save(folder/'expert-position-mask.json',mask)
        cfg=yaml.safe_load((folder/'task.yaml').read_text())
        cfg['garmi_task']['multi_object']['manifest']=inside(folder/'expert-bank.json')
        cfg['garmi_task']['kitchen']['mask']=inside(folder/'expert-position-mask.json')
        cfg['garmi_task']['kitchen']['files_sha256']['mask']=sha(folder/'expert-position-mask.json')
        rng=random.Random(2026101822)
        skipped=[]
        def sample(number,prefix):
            result=[]
            for j,ref in enumerate(mask['reference_ids']):
                eligible=[c for c in mask['cells'] if j in c['allowed_reference_indices']]
                if not eligible:
                    if ref not in skipped:skipped.append(ref)
                    continue
                for k in range(number):
                    requested='left' if k<round(number*.75) else 'right'
                    region_cells=[c for c in eligible if c['region']==requested]
                    cell=rng.choice(region_cells or eligible)
                    lo,hi=cell['bounds_map_xy']
                    x,y=[rng.uniform(lo[a]+1.e-5,hi[a]-1.e-5) for a in range(2)]
                    blo,bhi=cfg['garmi_task']['kitchen']['base_bounds_map_xy']
                    bx,by=[rng.uniform(blo[a],bhi[a]) for a in range(2)]
                    result.append(dict(id=f'{prefix}-{ref}-{k:03d}',object=name,reference=ref,
                        region=cell['region'],bowl_map_xy=[x,y],
                        robot_state=[by-6.35,.3-bx,.0259,1,0,0,0,0,0,0,0,0,0],
                        object_state=[y-6.35,.3-x,mask['bowl_origin_z_m'],2**-.5,0,0,-2**-.5,0,0,0,0,0,0]))
            return result
        evaluation=sample(100,'eval');audit=sample(10,'audit')
        keys=[tuple(c['robot_state']+c['object_state']) for c in evaluation+audit]
        assert len(set(keys))==len(keys)
        save(folder/'evaluation-cases.json',evaluation)
        save(folder/'expert-audit-cases.json',audit)
        cfg['evaluation']=dict(cases=inside(folder/'evaluation-cases.json'),cases_sha256=sha(folder/'evaluation-cases.json'))
        (folder/'expert-task.yaml').write_text(yaml.safe_dump(cfg))
        save(folder/'expert-preparation.json',dict(status='awaiting_user_confirmation',expert_trajectories_generated=0,
            audit_case_count=len(audit),evaluation_case_count=len(evaluation),
            references_blocked_by_geometry=skipped,
            source_diagnostic_bank_sha256=sha(folder/'bank.json'),
            source_diagnostic_mask_sha256=sha(folder/'position-mask.json'),
            note='Initial states only. No actions, rollouts, demonstrations, optimizer steps, or supervisor launch.'))
        print('EXPERT_INPUTS_PREPARED_NOT_EXECUTED',name,len(audit),skipped)


if __name__=='__main__':main()
