"""Verify PPO transfer/resume preserves frozen normalization and advances optimizer."""
import argparse,json,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--bc',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--stage',required=True);p.add_argument('--expected-update',type=int,required=True);p.add_argument('--out',required=True)
a=p.parse_args();stage=Path(a.stage);checkpoint=Path(a.checkpoint)
base=torch.load(a.bc,map_location='cpu',weights_only=False)
current=torch.load(checkpoint,map_location='cpu',weights_only=False)
resume=torch.load(checkpoint.with_suffix('.resume.pt'),map_location='cpu',weights_only=False)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==resume['checkpoint_sha256']
assert resume['update']==a.expected_update
for k,v in base['state_preprocessor'].items():assert torch.equal(v,current['state_preprocessor'][k]),k
assert all(torch.isfinite(v).all() for v in current['policy'].values())
assert any(not torch.equal(v,current['policy'][k]) for k,v in base['policy'].items())
progress=json.loads((stage/'progress.json').read_text());assert progress['status']=='complete' and progress['update']==a.expected_update
assert progress['frozen_state_normalization_verified']
assert 'Loss / Expert BC loss' in progress['last_update']['metrics']
assert resume['learning_config']['expert_bc']['coefficient']==3
assert resume['learning_config']['expert_bc']['state_normalization']=='frozen'
assert resume['learning_config']['garmi_task']['require_strict_success']
assert resume['learning_config']['garmi_task']['multi_object']['training_reset_pool']['split']=='train'
starts=sorted(stage.glob('initialization_*.json'));initial=json.loads(starts[-1].read_text())
if initial['mode']=='resume':assert initial['optimizer_state_entries']>0 and min(initial['optimizer_steps'])>0
assert current['optimizer']['state']
report=dict(status='passed',update=a.expected_update,normalization_unchanged=True,policy_updated=True,
            expert_bc_loss=progress['last_update']['metrics']['Loss / Expert BC loss'],
            optimizer_state_entries=len(current['optimizer']['state']),initialization=initial,
            checkpoint_sha256=resume['checkpoint_sha256'])
Path(a.out).write_text(json.dumps(report,indent=2));print('PPO_TRANSFER_AUDIT_PASSED',a.expected_update,flush=True)
