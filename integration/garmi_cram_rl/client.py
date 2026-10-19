"""Blocking skill calls from a CRAM node whose ROS executor is already spinning."""
import json
import threading
import time
import uuid

from std_msgs.msg import String


class GraspClient:
    def __init__(self, node):
        self.publisher = node.create_publisher(String, '/fame/rl_grasp/request', 20)
        self.state = None
        self.received = 0.
        self.replies = {}
        self.condition = threading.Condition()
        self.state_subscription = node.create_subscription(String, '/fame/rl_grasp/state', self.on_state, 1)
        self.reply_subscription = node.create_subscription(String, '/fame/rl_grasp/reply', self.on_reply, 20)

    def on_state(self, message):
        with self.condition:
            self.state, self.received = json.loads(message.data), time.monotonic()
            self.condition.notify_all()

    def on_reply(self, message):
        value = json.loads(message.data)
        with self.condition:
            self.replies[value['rpc_id']] = value
            self.condition.notify_all()

    def snapshot(self, timeout=15.):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.state is None or time.monotonic() - self.received > .5:
                if time.monotonic() >= deadline:
                    raise TimeoutError('RL bridge has no fresh simulator state')
                self.condition.wait(.1)
            if self.state['fault'] is not None:
                raise RuntimeError(self.state['fault'])
            return self.state

    def call(self, command, request_id, timeout=10., **values):
        rpc_id = str(uuid.uuid4())
        message = String(data=json.dumps(dict(rpc_id=rpc_id, command=command, request_id=request_id, **values)))
        deadline, sent = time.monotonic() + timeout, -float('inf')
        with self.condition:
            while rpc_id not in self.replies:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(f'RL bridge did not acknowledge {command}')
                if now - sent >= .5:
                    self.publisher.publish(message)
                    sent = now
                self.condition.wait(.05)
            reply = self.replies.pop(rpc_id)
        if not reply['ok']:
            raise RuntimeError(reply['error'])
        return reply

    def grasp(self, reference=None, timeout=180.):
        request_id = str(uuid.uuid4())
        self.call('start', request_id, reference=reference)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                self.publisher.publish(String(data=json.dumps(dict(rpc_id=str(uuid.uuid4()),
                    command='heartbeat', request_id=request_id))))
                state = self.snapshot()
                if state['request_id'] == request_id and state['owner'] == 'holding':
                    if not state['result']['success']:
                        raise RuntimeError(f"RL grasp failed: {state['result']['reason']}")
                    # Return only after the simulator confirms exclusive CRAM ownership.
                    self.call('release', request_id)
                    return state
                with self.condition:
                    self.condition.wait(.1)
            raise TimeoutError('RL grasp did not finish in the wall-clock budget')
        except BaseException as error:
            # Cancellation stops and holds; it does not reset physics or open the fingers.
            try:
                self.call('cancel', request_id)
            except (RuntimeError, TimeoutError) as cancellation_error:
                error.add_note(f'Cancellation acknowledgement unavailable: {cancellation_error}')
            raise
