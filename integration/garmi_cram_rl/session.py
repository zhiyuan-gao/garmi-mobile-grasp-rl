"""Exclusive controller ownership; terminal states hold until explicitly released."""
from dataclasses import dataclass


@dataclass
class Session:
    backend: object
    heartbeat_timeout: float = 2.0
    owner: str = 'cram'
    request_id: str | None = None
    last_heartbeat: float = 0.0
    result: dict | None = None

    def start(self, request_id, reference, now):
        if not request_id:
            raise ValueError('A request id is required')
        if request_id == self.request_id:
            return self.status()
        if self.owner != 'cram':
            raise RuntimeError('Another grasp owns the robot')
        self.backend.admit(reference)
        self.backend.begin(reference)
        self.owner, self.request_id = 'rl', request_id
        self.last_heartbeat, self.result = now, None
        return self.status()

    def heartbeat(self, request_id, now):
        self.require_request(request_id)
        self.last_heartbeat = now

    def require_request(self, request_id):
        if request_id != self.request_id or self.owner == 'cram':
            raise ValueError('Request does not own this grasp')

    def finish(self, reason):
        self.backend.hold()
        self.owner = 'holding'
        self.result = {'success': reason == 'success', 'reason': reason}

    def tick(self, now):
        if self.owner != 'rl':
            return
        if now - self.last_heartbeat > self.heartbeat_timeout:
            self.finish('client_timeout')
            return
        reason = self.backend.step_policy()
        if reason is not None:
            self.finish(reason)

    def cancel(self, request_id):
        self.require_request(request_id)
        if self.owner == 'rl':
            self.finish('cancelled')

    def release(self, request_id):
        self.require_request(request_id)
        if self.owner == 'rl':
            raise RuntimeError('Cancel or finish the grasp before returning control')
        self.backend.return_to_cram()
        self.owner = 'cram'
        return self.status()

    def status(self):
        return {'owner': self.owner, 'request_id': self.request_id, 'result': self.result}
