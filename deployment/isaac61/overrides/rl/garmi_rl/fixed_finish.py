"""Policy approach followed by a physical, deterministic close/lift attempt.

The policy still acts every control tick, but its actions have no effect during
the finishing phases. Phase, timer and controller target are observed so this
is an ordinary MDP with action-independent dynamics during those phases. There
are no teleports, object attachments or success labels based only on alignment.
"""
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude, skew_symmetric_matrix
from .actions import PlanarVelocityAction, PlanarVelocityActionCfg

PHASES = ("approach", "settle", "close", "lift", "hold")
DEFAULTS = dict(alignment_distance=.015, alignment_angle=.15, alignment_seconds=.10,
                approach_seconds=6., settle_seconds=.5, close_seconds=2.,
                lift_seconds=3., lift_distance=.12, hold_seconds=3.)


class FixedFinish:
    def __init__(self, env):
        self.env = env
        self.settings = env.cfg.fixed_finish
        n, dev = env.num_envs, env.device
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase_steps = torch.zeros_like(self.phase)
        self.aligned_steps = torch.zeros_like(self.phase)
        self.trigger = torch.zeros_like(self.phase)  # 1=alignment; 2=approach deadline
        self.handover_distance = torch.zeros(n, device=dev)
        self.handover_angle = torch.zeros(n, device=dev)
        self.position = torch.zeros(n, 3, device=dev)
        self.rotation = torch.zeros(n, 4, device=dev)
        self.rotation[:, 0] = 1.
        self.target_position = self.position.clone()
        self.target_rotation = self.rotation.clone()
        robot = env.scene["robot"]
        self.arm_ids = robot.find_joints("right_fr3_joint[1-7]")[0]
        self.joint_target = robot.data.default_joint_pos[:, self.arm_ids].clone()
        self.controller = DifferentialIKController(DifferentialIKControllerCfg(
            command_type="pose", ik_method="dls"), n, dev)

    def ticks(self, key):
        return round(self.settings[key] / self.env.step_dt)

    def reset(self, env_ids):
        for value in (self.phase, self.phase_steps, self.aligned_steps, self.trigger):
            value[env_ids] = 0
        self.handover_distance[env_ids] = 0
        self.handover_angle[env_ids] = 0
        self.position[env_ids] = 0
        self.rotation[env_ids] = 0
        self.rotation[env_ids, 0] = 1.
        self.target_position[env_ids] = 0
        self.target_rotation[env_ids] = self.rotation[env_ids]

    def advance(self):
        # Called once per policy step by the first action term, before physics.
        from . import mdp
        env = self.env
        hp, hq, gp, gq = mdp.geometry(env)
        self.phase_steps += 1
        approaching = self.phase == 0
        aligned = ((gp-hp).norm(dim=-1) < self.settings["alignment_distance"]) & (
            quat_error_magnitude(hq, gq) < self.settings["alignment_angle"])
        self.aligned_steps = torch.where(approaching & aligned, self.aligned_steps + 1, 0)
        ready = self.aligned_steps >= self.ticks("alignment_seconds")
        deadline = self.phase_steps >= self.ticks("approach_seconds")
        enter = approaching & (ready | deadline)
        self.position[enter] = hp[enter]
        self.rotation[enter] = hq[enter]
        self.trigger[enter] = torch.where(ready[enter], 1, 2)
        self.handover_distance[enter] = (gp-hp).norm(dim=-1)[enter]
        self.handover_angle[enter] = quat_error_magnitude(hq,gq)[enter]
        # Descending order prevents two phase transitions in the same tick.
        for phase, duration in ((3, "lift_seconds"), (2, "close_seconds"), (1, "settle_seconds")):
            switch = (self.phase == phase) & (self.phase_steps >= self.ticks(duration))
            self.phase[switch] += 1
            self.phase_steps[switch] = 0
        self.phase[enter] = 1
        self.phase_steps[enter] = 0
        self.target_position[:] = self.position
        self.target_rotation[:] = self.rotation
        fraction = (self.phase_steps.float() / self.ticks("lift_seconds")).clamp(0., 1.)
        fraction = torch.where(self.phase == 3, fraction, (self.phase >= 4).float())
        self.target_position[:, 2] += self.settings["lift_distance"] * fraction
        self.target_position[approaching & ~enter] = gp[approaching & ~enter]
        self.target_rotation[approaching & ~enter] = gq[approaching & ~enter]
        # Use the same hand-offset/world-frame DLS as the verified diagnostic.
        # Skip the batched solve while all environments are still approaching.
        if not torch.any(self.phase > 0):
            return
        robot = env.scene["robot"]
        jac = robot.root_physx_view.get_jacobians()[:, env.hand_id, :,
            [i+6 for i in self.arm_ids]].clone()
        offset = hp - robot.data.body_pos_w[:, env.hand_id]
        jac[:, :3, :] -= torch.bmm(skew_symmetric_matrix(offset), jac[:, 3:, :])
        self.controller.set_command(torch.cat((self.target_position, self.target_rotation), dim=-1))
        q = self.controller.compute(hp, hq, jac, robot.data.joint_pos[:, self.arm_ids])
        limits = robot.data.soft_joint_pos_limits[:, self.arm_ids]
        self.joint_target[:] = q.clamp(limits[:, :, 0], limits[:, :, 1])


class FinishBaseAction(PlanarVelocityAction):
    """Policy base velocity until handover; then stop with the existing acceleration limit."""
    def process_actions(self, actions):
        finish = self._env.fixed_finish
        finish.advance()
        super().process_actions(actions)
        self._processed_actions[finish.phase > 0] = 0.


class FinishArmAction(JointPositionAction):
    """Policy arm target until handover, followed by the deterministic IK target."""
    def process_actions(self, actions):
        super().process_actions(actions)
        finish = self._env.fixed_finish
        scripted = finish.phase > 0
        self._processed_actions[scripted] = finish.joint_target[scripted]


class FinishGripperAction(ActionTerm):
    """Zero-dimensional action term: open during approach/settle, close afterwards."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids = self._asset.find_joints("right_fr3_finger_joint.*")[0]
        self._raw_actions = torch.zeros(env.num_envs, 0, device=env.device)
        self._processed_actions = torch.full((env.num_envs, 2), .04, device=env.device)
        env.fixed_finish = FixedFinish(env)

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._processed_actions[:] = torch.where(self._env.fixed_finish.phase[:, None] >= 2, 0., .04)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        self._processed_actions[env_ids] = .04


@configclass
class FinishBaseActionCfg(PlanarVelocityActionCfg):
    class_type: type = FinishBaseAction


@configclass
class FinishArmActionCfg(JointPositionActionCfg):
    class_type: type = FinishArmAction


@configclass
class FinishGripperActionCfg(ActionTermCfg):
    class_type: type = FinishGripperAction
