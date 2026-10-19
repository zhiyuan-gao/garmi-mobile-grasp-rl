"""Giskard can command exactly the joints present in the trained articulation."""
ACTIVE_JOINTS = [f'right_fr3_joint{i}' for i in range(1, 8)] + [
    'right_fr3_finger_joint1', 'right_fr3_finger_joint2']
ACTIVE_VELOCITY_TOPIC = '/fame/rl_grasp/active_joint_velocity'
LEFT_JOINTS = [f'left_fr3_joint{i}' for i in range(1, 8)] + [
    'left_fr3_finger_joint1', 'left_fr3_finger_joint2']


def make_interface(dual_arm=False):
    from cram_vrb_lab.robots.garmi.giskard_config import GarmiSimInterface
    active = ACTIVE_JOINTS + LEFT_JOINTS if dual_arm else ACTIVE_JOINTS

    class TrainedGarmiInterface(GarmiSimInterface):
        def add_joint_velocity_group_controller(self, cmd_topic, connections, **kwargs):
            if not set(active).issubset(connections):
                raise ValueError('The native GARMI controller does not contain the trained joints')
            return super().add_joint_velocity_group_controller(
                cmd_topic=ACTIVE_VELOCITY_TOPIC, connections=active, **kwargs)

    return TrainedGarmiInterface()
