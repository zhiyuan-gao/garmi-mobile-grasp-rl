"""Sequential recordings of six failed frozen cases and two successful controls."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/cup4-preppo'
BASE='/workspace/runs/cup4-preppo'


def main():
    results=[]
    for index in (70,18,27,41,51,59,21,74):
        name=f'c{index:04d}'
        destination=RUN/'failure-review'/name
        if not (destination/'summary.json').exists():
            if destination.exists():
                raise RuntimeError(f'Incomplete recording requires review: {destination}')
            command=['docker','exec','-e','PYTHONPATH=/workspace/rl','-e','OPENBLAS_NUM_THREADS=1',
                     '-e','OMP_NUM_THREADS=1','garmi-cup-rl','/isaac-sim/python.sh',
                     '/workspace/rl/diagnose_full_rollout_physics.py',
                     '--config',BASE+'/task.yaml','--cases',BASE+'/expert-audit-cases.json',
                     '--plans',BASE+'/full-rollout-80/plans.json','--base-feedback','--retime','--contacts',
                     '--case-index',str(index),'--enable_cameras','--headless',
                     '--out',BASE+'/failure-review/'+name]
            with (RUN/f'failure-review-{name}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        value=json.loads((destination/'summary.json').read_text())
        assert value['status']=='complete' and (destination/'replay.mp4').stat().st_size>1000
        result=dict(case=index,reason=value['results'][0]['physics']['reason'])
        results.append(result)
        (RUN/'failure-review-progress.json').write_text(json.dumps(dict(completed=results,total=8),indent=2))
        print('RECORDED',result,flush=True)


if __name__=='__main__':main()
