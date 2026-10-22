"""Interrupt a blocked native motion without waiting for its result mailbox."""
from contextlib import contextmanager
import signal


@contextmanager
def motion_deadline(seconds, label):
    """Raise a timeout on the demo's main thread, including during blocking waits."""
    def expired(signum, frame):
        raise TimeoutError(f'{label} exceeded {seconds:g} seconds')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
