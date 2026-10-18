"""Refit training-only input statistics and rebase the first affine layer."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from gymnasium.spaces import Box
from skrl.utils.model_instantiators.torch import shared_model
from garmi_rl.expert_learning import load_demonstrations, policy_mean

p=argparse.ArgumentParser()
for name in ('index','checkpoint','config','out'):p.add_argument('--'+name,required=True)
args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
data=load_demonstrations(args.index,'cpu')
saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
old=saved['state_preprocessor'];states=data['full_states'].double()
new=copy.deepcopy(old);new['running_mean']=states.mean(0)
new['running_variance']=states.var(0,unbiased=False).clamp_min(1.e-4)
new['current_count']=torch.full_like(old['current_count'],len(states))
del states
old_mean=old['running_mean'].float();new_mean=new['running_mean'].float()
old_std=old['running_variance'].float().sqrt()+1.e-8
new_std=new['running_variance'].float().sqrt()+1.e-8
old_policy=saved['policy'];weights=old_policy['net_container.0.weight'].clone()
rebased=copy.deepcopy(old_policy)
rebased['net_container.0.weight']=weights*(new_std/old_std)[None,:]
rebased['net_container.0.bias']=old_policy['net_container.0.bias']+weights@((new_mean-old_mean)/old_std)
payload={k:copy.deepcopy(v) for k,v in saved.items() if k!='optimizer'}
payload.update(policy=rebased,value=copy.deepcopy(rebased),state_preprocessor=new)
x=data['states'];z_old=(x-old_mean)/old_std;z_new=(x-new_mean)/new_std
geometry=list(range(24,37))+list(range(55,62))
unclipped=((z_old.abs()<=5).all(1)&(z_new.abs()<=5).all(1)).nonzero().flatten()
assert len(unclipped)>=128,'Insufficient overlap to verify the normalization conversion'
selected=unclipped[torch.linspace(0,len(unclipped)-1,min(4096,len(unclipped))).long()]
cfg=yaml.safe_load(Path(args.config).read_text());models=copy.deepcopy(cfg['models']);models.pop('separate')
roles=list(models);structure=[models[r].pop('class') for r in roles]
def model(state):
    obj=shared_model(observation_space=Box(-np.inf,np.inf,(64,)),action_space=Box(-np.inf,np.inf,(10,)),
        device='cpu',structure=structure,roles=roles,parameters=[models[r] for r in roles])
    for r in roles:obj.init_state_dict(r)
    obj.load_state_dict(state);return obj
with torch.no_grad():
    a=policy_mean(model(old_policy),z_old[selected]);b=policy_mean(model(rebased),z_new[selected])
    error=float((a-b).abs().max())
assert error<1.e-4,error
per_reference={}
for ref in np.unique(data['references']):
    ids=data['references']==ref
    per_reference[str(ref)]=dict(samples=int(ids.sum()),
        old_geometry_clipped_fraction=float((z_old[ids][:,geometry].abs()>5).any(1).float().mean()),
        new_geometry_clipped_fraction=float((z_new[ids][:,geometry].abs()>5).any(1).float().mean()))
path=out/'actor-init.pt';torch.save(payload,path)
report=dict(status='passed',training_index=args.index,source_checkpoint=args.checkpoint,
    source_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
    checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    statistics_source='Training trajectories only, all phases; no validation/test states',
    normalizer_count=float(new['current_count']),per_reference=per_reference,
    first_layer_rebased=True,unclipped_policy_max_error=error,verified_states=len(selected),
    clipped_states_note='Exact equivalence applies where neither normalizer clips. Newly unclipped states can change; BC follows.',
    optimizer='Not transferred: subsequent BC uses fresh Adam')
(out/'summary.json').write_text(json.dumps(report,indent=2));print('NORMALIZATION_40REF_DONE',json.dumps(report),flush=True)
