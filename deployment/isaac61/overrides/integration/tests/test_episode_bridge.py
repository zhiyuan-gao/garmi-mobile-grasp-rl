"""Episode reset clears stale ownership and acknowledges retried RPCs only once."""
import io
import json
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from std_msgs.msg import String

from garmi_cram_rl.ros_bridge import Bridge
from garmi_cram_rl.session import Session


def test_reset_recovers_fault_and_deduplicates_request(tmp_path):
    resets, replies = [], []
    backend = SimpleNamespace(reset_episode=lambda: resets.append('reset'),
                              snapshot=lambda: {'reference': None, 'phase': 0})
    bridge = Bridge.__new__(Bridge)
    bridge.backend = backend
    bridge.session = Session(backend, owner='holding', request_id='old', result={'success': False})
    bridge.fault = 'old command failure'
    bridge.accept_after = 0
    bridge.replies = {'old': 'stale reply'}
    previous_log = bridge.log = io.StringIO()
    bridge.reply_pub = SimpleNamespace(publish=lambda msg: replies.append(json.loads(msg.data)))
    bridge.reset_handler = lambda path: bridge.reset_episode(tmp_path / 'events.jsonl')
    message = String(data=json.dumps(dict(command='reset_episode', rpc_id='reset',
                                          request_id='new', output=str(tmp_path))))
    try:
        bridge.request(message)
        bridge.request(message)
        assert resets == ['reset']
        assert len(replies) == 2 and replies[0] == replies[1]
        assert replies[0]['ok'] and replies[0]['owner'] == 'cram'
        assert bridge.session.request_id is None and bridge.session.result is None
        assert bridge.fault is None and bridge.accept_after > 0
        assert list(bridge.replies) == ['reset']
        assert previous_log.closed
        events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
        assert [event['event'] for event in events] == ['episode_reset', 'request']
    finally:
        bridge.log.close()


def test_reset_is_disabled_unless_explicitly_enabled():
    bridge = Bridge.__new__(Bridge)
    bridge.reset_handler = None
    bridge.replies = {}
    bridge.log = io.StringIO()
    replies = []
    bridge.reply_pub = SimpleNamespace(publish=lambda msg: replies.append(json.loads(msg.data)))
    bridge.request(String(data=json.dumps(dict(command='reset_episode', rpc_id='reset'))))
    assert not replies[0]['ok'] and 'not enabled' in replies[0]['error']
