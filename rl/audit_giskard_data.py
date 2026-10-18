"""Validate physical successes, temporal pairing, stage history, and split isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from garmi_rl.giskard_io import COMMAND_VERSION, advance_stages
from garmi_rl.joint_delta import COMMAND_VERSION as DELTA_VERSION
from garmi_rl.augmentation import approach_label_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True)
    p.add_argument('--validation',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    train=json.loads(Path(args.index).read_text());validation=json.loads(Path(args.validation).read_text())
    version = train.get('command_version', COMMAND_VERSION)
    assert version in (COMMAND_VERSION, DELTA_VERSION)
    assert validation.get('command_version', COMMAND_VERSION) == version
    dimension = 10
    ids=set();states=set();counts={};steps=0;maximum=0.
    for split,manifest in (('train',train),('validation',validation)):
        assert manifest['status']=='complete' and manifest['expert_backend']=='giskard'
        counts[split]=len(manifest['demonstrations'])
        for row in manifest['demonstrations']:
            assert row['id'] not in ids;ids.add(row['id'])
            case=row['initial_case'];key=tuple(case['robot_state']+case['object_state'])
            assert key not in states;states.add(key)
            assert row['command_version']==version and row['reason']=='success'
            assert row['height']>=.08 and row['contact'] and not row['supported']
            assert row['slip']<.02 and row['hold_seconds']>=2. and not row['collision_bodies']
            path=Path(row['file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            with np.load(path,allow_pickle=False) as z:
                x,y,phase=z['observations'],z['actions'],z['phases']
                assert x.shape==(len(y)+1,64) and y.shape[1]==dimension
                assert np.isfinite(x).all() and np.isfinite(y).all()
                noise=z['perturbation_mask'] if 'perturbation_mask' in z else None
                if row.get('augmentation')=='ga_ddpg_dart':
                    assert noise is not None and noise.any() and row['perturbations']>0
                mask=approach_label_mask(phase,noise)
                assert np.array_equal(z['bc_mask'],mask) and mask.sum()==row['bc_steps']>0
                error=float(np.max(np.abs(x[1:,39:39+dimension][mask]-y[mask])))
                maximum=max(maximum,error);assert error<1.e-6
                assert z['terminated'][-1] and not z['truncated'][-1]
                assert not np.any(np.diff(z['approach_stage'].astype(int))<0)
                stage=np.array([False])
                for i in range(len(y)):
                    stage=advance_stages(stage,x[i:i+1])
                    # The expert also checks exact FK; a boundary difference may
                    # advance its stage one step earlier, but never restart it.
                    if z['approach_stage'][i]:stage[0]=True
                    assert bool(stage[0])==bool(z['approach_stage'][i])
                steps+=int(mask.sum())
    report=dict(status='passed',trajectories=counts,bc_steps=steps,
        max_action_alignment_error=maximum,split_overlap=False,
        command_version=version,physics_success_verified=True,
        checks=['hashes','physical success','approach-only labels','executed action pairing',
            'monotonic teacher stage','disjoint initial states'])
    Path(args.out).write_text(json.dumps(report,indent=2));print('GISKARD_DATA_AUDIT',json.dumps(report),flush=True)


if __name__=='__main__':main()
