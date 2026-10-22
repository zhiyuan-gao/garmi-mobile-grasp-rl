"""Planar root velocity servo; arm/gripper use standard Isaac Lab actions."""
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply


class PlanarVelocityAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.velocity = torch.zeros_like(self._raw_actions)
        self.scale = torch.tensor(cfg.scale, device=self.device)
        self.accel = torch.tensor(cfg.acceleration, device=self.device)
        self.command = torch.zeros(self.num_envs, 6, device=self.device)

    @property
    def action_dim(self):
        return 3

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions.clamp(-1., 1.) * self.scale

    def apply_actions(self):
        dv = self.accel * self._env.physics_dt
        self.velocity += torch.clamp(self._processed_actions - self.velocity, -dv, dv)
        local = torch.zeros_like(self.velocity)
        local[:, :2] = self.velocity[:, :2]
        self.command.zero_()
        self.command[:, :2] = quat_apply(self._asset.data.root_quat_w, local)[:, :2]
        self.command[:, 5] = self.velocity[:, 2]
        self._asset.write_root_velocity_to_sim(self.command)

    def reset(self, env_ids=None):
        self._raw_actions[env_ids] = 0
        self._processed_actions[env_ids] = 0
        self.velocity[env_ids] = 0


@configclass
class PlanarVelocityActionCfg(ActionTermCfg):
    class_type: type = PlanarVelocityAction
    scale: tuple = (.15, .15, .4)
    acceleration: tuple = (.3, .3, .8)
