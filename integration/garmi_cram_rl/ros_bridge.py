"""Existing GARMI ROS commands plus an exclusive RL grasp request channel."""
import json
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, String

from cram_vrb_lab.robots.garmi.joints import (
    CONTROLLED_JOINTS, VELOCITY_CMD_TOPIC, CMD_VEL_TOPIC, GRIPPER_CMD_TOPIC,
    JOINT_STATES_TOPIC, ODOM_TOPIC, BASE_LINK_HEIGHT,
)
from garmi_cram_rl.session import Session
from garmi_cram_rl.robot_interface import ACTIVE_JOINTS, ACTIVE_VELOCITY_TOPIC


class Bridge(Node):
    def __init__(self, backend, output):
        super().__init__('garmi_rl_skill_bridge')
        self.backend, self.session = backend, Session(backend)
        self.log = output.open('a', buffering=1)
        self.replies = {}
        self.fault = None
        self.accept_after = 0.
        self.create_subscription(Float64MultiArray, VELOCITY_CMD_TOPIC, self.joint_command, 1)
        self.create_subscription(Float64MultiArray, ACTIVE_VELOCITY_TOPIC, self.active_joint_command, 1)
        self.create_subscription(Twist, CMD_VEL_TOPIC, self.base_command, 1)
        self.create_subscription(Float64, GRIPPER_CMD_TOPIC, self.gripper_command, 1)
        self.create_subscription(String, '/fame/rl_grasp/request', self.request, 20)
        self.create_subscription(String, '/fame/task_event', self.task_event, 20)
        self.state_pub = self.create_publisher(String, '/fame/rl_grasp/state', 1)
        self.reply_pub = self.create_publisher(String, '/fame/rl_grasp/reply', 20)
        self.joint_pub = self.create_publisher(JointState, JOINT_STATES_TOPIC, 10)
        self.odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self.object_pub = self.create_publisher(String, '/cram_vrb_lab/scene_sync_poses', 10)

    def task_event(self, message):
        event = json.loads(message.data)
        self.log.write(json.dumps({'event': 'task_event', 'task': event}) + '\n')
        if event.get('event') == 'camera':
            self.backend.camera(event['eye'], event['target'])

    def classical(self):
        return self.session.owner == 'cram' and time.monotonic() >= self.accept_after and self.fault is None

    def command_fault(self, error):
        self.fault = str(error)
        self.backend.clear_classical_commands()
        self.log.write(json.dumps({'event': 'command_fault', 'error': self.fault}) + '\n')

    def joint_command(self, message):
        if self.classical():
            try:
                self.backend.set_joint_velocity(CONTROLLED_JOINTS, message.data, time.monotonic())
            except ValueError as error:
                self.command_fault(error)

    def active_joint_command(self, message):
        if self.classical():
            try:
                names = getattr(self.backend, 'active_joint_names', ACTIVE_JOINTS)
                self.backend.set_joint_velocity(names, message.data, time.monotonic())
            except ValueError as error:
                self.command_fault(error)

    def base_command(self, message):
        if self.classical():
            try:
                self.backend.set_base_velocity([message.linear.x, message.linear.y, message.angular.z],
                                               time.monotonic())
            except ValueError as error:
                self.command_fault(error)

    def gripper_command(self, message):
        if self.classical():
            try:
                self.backend.set_gripper(message.data)
            except ValueError as error:
                self.command_fault(error)

    def request(self, message):
        request = json.loads(message.data)
        rpc_id, command = request['rpc_id'], request['command']
        if command == 'heartbeat':
            if request.get('request_id') == self.session.request_id and self.session.owner != 'cram':
                self.session.heartbeat(request['request_id'], time.monotonic())
            return
        if rpc_id in self.replies:
            self.reply_pub.publish(String(data=self.replies[rpc_id]))
            return
        try:
            if self.fault is not None:
                raise RuntimeError(self.fault)
            if command == 'start':
                self.session.start(request['request_id'], request.get('reference'), time.monotonic())
            elif command == 'cancel':
                self.session.cancel(request['request_id'])
            elif command == 'release':
                self.session.release(request['request_id'])
                self.accept_after = time.monotonic() + .1
            else:
                raise ValueError('Unknown skill command')
            reply = dict(rpc_id=rpc_id, ok=True, **self.session.status())
        except (ValueError, RuntimeError) as error:
            reply = dict(rpc_id=rpc_id, ok=False, error=str(error))
        self.replies[rpc_id] = json.dumps(reply)
        self.reply_pub.publish(String(data=self.replies[rpc_id]))
        self.log.write(json.dumps({'event': 'request', 'request': request, 'reply': reply}) + '\n')

    def step(self):
        was_running = self.session.owner == 'rl'
        self.session.tick(time.monotonic())
        if not was_running:
            self.backend.step_classical(time.monotonic(), holding=self.session.owner == 'holding' or self.fault is not None)
        if was_running and self.session.owner == 'holding':
            self.log.write(json.dumps({'event': 'grasp_result', **self.session.status(),
                                      'snapshot': self.backend.snapshot()}) + '\n')

    def publish(self):
        state = self.backend.snapshot()
        self.state_pub.publish(String(data=json.dumps(dict(**self.session.status(), snapshot=state, fault=self.fault))))
        joints = JointState()
        joints.header.stamp = self.get_clock().now().to_msg()
        joints.name = list(state['joints'])
        joints.position = list(state['joints'].values())
        joints.velocity = [state['velocities'].get(name, 0.) for name in joints.name]
        self.joint_pub.publish(joints)
        odom = Odometry()
        odom.header.stamp = joints.header.stamp
        odom.header.frame_id, odom.child_frame_id = 'odom', 'base_link'
        x, y, z = state['base_map_position']
        odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
        odom.pose.pose.position.z = z - BASE_LINK_HEIGHT
        q = state['base_map_quaternion_wxyz']
        odom.pose.pose.orientation.w, odom.pose.pose.orientation.x = q[:2]
        odom.pose.pose.orientation.y, odom.pose.pose.orientation.z = q[2:]
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z = state['base_velocity']
        self.odom_pub.publish(odom)
        q = state['object_map_quaternion_wxyz']
        self.object_pub.publish(String(data=json.dumps({'graspgen_bowl_stl': {
            'position': state['object_map_position'], 'orientation': [*q[1:], q[0]]}})))

    def destroy_node(self):
        self.backend.clear_classical_commands()
        self.log.close()
        return super().destroy_node()
