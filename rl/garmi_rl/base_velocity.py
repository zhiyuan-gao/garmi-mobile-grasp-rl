"""Version the velocity reference point without changing existing PPO profiles."""


def velocity_at_link(task):
    frame = task.get('base_velocity_frame', 'root_com_legacy')
    if frame not in ('root_com_legacy', 'root_link'):
        raise ValueError(f'Unknown base velocity frame: {frame}')
    return frame == 'root_link'


def planning_root_state(env):
    robot = env.scene['robot']
    if env.cfg.actions.base.velocity_at_link:
        return robot.data.root_link_state_w
    return robot.data.root_state_w
