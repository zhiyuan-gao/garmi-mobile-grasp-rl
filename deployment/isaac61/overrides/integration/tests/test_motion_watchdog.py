import signal
import threading

import pytest

from motion_watchdog import motion_deadline


def test_blocking_motion_is_interrupted_and_alarm_is_restored():
    original = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError, match='blocked motion'):
        with motion_deadline(.03, 'blocked motion'):
            threading.Event().wait(1)
    assert signal.getsignal(signal.SIGALRM) == original
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_completed_motion_does_not_leave_a_pending_timeout():
    with motion_deadline(.03, 'completed motion'):
        pass
    threading.Event().wait(.06)
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
