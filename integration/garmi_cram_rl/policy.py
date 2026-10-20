"""CPU-only deterministic actor with the frozen skrl input normalizer."""
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        if checkpoint.get('action_contract', {}).get('name') != 'base_velocity_joint_delta_v1':
            raise ValueError('Wrong checkpoint action contract')
        weights = checkpoint['policy']
        indices = sorted(int(key.split('.')[1]) for key in weights
                         if key.startswith('net_container.') and key.endswith('.weight'))
        layers = []
        for index in indices:
            weight, bias = weights[f'net_container.{index}.weight'], weights[f'net_container.{index}.bias']
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            layers.extend((layer, nn.ELU()))
        layer = nn.Linear(weights['policy_layer.weight'].shape[1], 10)
        layer.weight.data.copy_(weights['policy_layer.weight'])
        layer.bias.data.copy_(weights['policy_layer.bias'])
        self.network = nn.Sequential(*layers, layer)
        normalizer = checkpoint['state_preprocessor']
        self.register_buffer('mean', normalizer['running_mean'].float().clone())
        self.register_buffer('variance', normalizer['running_variance'].float().clone())

    def forward(self, observation):
        normalized = ((observation - self.mean) / (torch.sqrt(self.variance) + 1.e-8)).clamp(-5., 5.)
        # Evaluation uses mean_actions. Preserve raw means in action history;
        # the original action terms perform their own physical clipping.
        return self.network(normalized)


class Policy:
    def __init__(self, bundle):
        torch.set_num_threads(1)
        self.actor = torch.jit.load(str(bundle.path / 'actor.ts'), map_location='cpu').eval()

    @torch.inference_mode()
    def __call__(self, observation):
        value = torch.as_tensor(observation, device='cpu', dtype=torch.float32).reshape(1, 64)
        if not torch.isfinite(value).all():
            raise ValueError('Invalid observation')
        result = self.actor(value)
        if result.shape != (1, 10) or not torch.isfinite(result).all():
            raise ValueError('Invalid policy command')
        return result
