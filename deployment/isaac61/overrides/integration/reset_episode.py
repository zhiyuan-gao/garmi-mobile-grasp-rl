"""Reset the owned simulator after the previous task and planner have exited."""
import argparse
from pathlib import Path
import threading
import uuid

import rclpy
from rclpy.executors import SingleThreadedExecutor

from garmi_cram_rl.client import GraspClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('garmi_episode_reset')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    client = GraspClient(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        client.call('reset_episode', str(uuid.uuid4()), timeout=45., output=str(args.output))
        print('EPISODE_RESET_COMPLETE', flush=True)
    finally:
        executor.shutdown()
        thread.join(timeout=3)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
