"""Direct relative joint position actions with the existing fixed finish controller."""
from isaaclab.utils import configclass
from .fixed_finish import FinishArmAction, FinishArmActionCfg
from .joint_delta import decode_joint_delta


class JointDeltaArmAction(FinishArmAction):
    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._processed_actions[ids] = self._asset.data.default_joint_pos[ids][:, self._joint_ids]

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        measured = self._asset.data.joint_pos[:, self._joint_ids]
        self._processed_actions[:] = decode_joint_delta(measured, actions)
        finish = self._env.fixed_finish
        scripted = finish.phase > 0
        self._processed_actions[scripted] = finish.joint_target[scripted]


@configclass
class JointDeltaArmActionCfg(FinishArmActionCfg):
    class_type: type = JointDeltaArmAction
