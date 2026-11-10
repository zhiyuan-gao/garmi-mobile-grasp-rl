"""Single-scene optimizations that retain every contact sample and history frame."""
import types


def optimize_contact_updates(scene):
    """Avoid a CUDA->CPU nonzero/index check for every always-updated sensor.

    Valid only for one environment, zero update period and nonempty history:
    SensorBase.update must update all environments on every call in this case.
    The original force acquisition/history implementation still runs unchanged.
    """
    from isaaclab.sensors import ContactSensor
    installed=[]
    for name,sensor in scene.sensors.items():
        if not isinstance(sensor,ContactSensor):
            continue
        if sensor._num_envs!=1 or sensor.cfg.update_period!=0 or sensor.cfg.history_length<=0:
            continue
        if getattr(sensor,'_single_scene_fast_update',False):
            continue
        install_contact_update(sensor)
        installed.append(name)
    return installed


def install_contact_update(sensor):
    original=dict(update=sensor.update, buffers=sensor._update_outdated_buffers, reset=sensor.reset)
    sensor._fast_dirty=True

    def buffers(self):
        if not self._fast_dirty:
            return
        # A one-element CPU sequence becomes slice(None) in the original method.
        # It avoids GPU nonzero() and indexed timestamp reads/writes.
        self._update_buffers_impl((0,))
        self._timestamp_last_update.copy_(self._timestamp)
        self._is_outdated.fill_(False)
        self._fast_dirty=False

    def update(self,dt,force_recompute=False):
        self._timestamp.add_(dt)
        self._fast_dirty=True
        self._update_outdated_buffers()

    def reset(self,env_ids=None):
        original['reset'](env_ids)
        self._fast_dirty=True

    sensor._update_outdated_buffers=types.MethodType(buffers,sensor)
    sensor.update=types.MethodType(update,sensor)
    sensor.reset=types.MethodType(reset,sensor)
    sensor._single_scene_fast_update=True
    sensor._original_contact_update=original


def restore_contact_updates(scene):
    for sensor in scene.sensors.values():
        original=getattr(sensor,'_original_contact_update',None)
        if original:
            sensor.update=original['update']
            sensor._update_outdated_buffers=original['buffers']
            sensor.reset=original['reset']
            sensor._single_scene_fast_update=False
            del sensor._original_contact_update


class RuntimeMetrics:
    """Small periodic wall-time/simulation-time/RSS log for the local demo."""
    def __init__(self, path):
        import time
        self.stream = path.open('w', buffering=1)
        self.started = self.last_time = time.monotonic()
        self.last_step = None
        self.rss_mb = 0.

    def sample(self, backend, force=False):
        import json
        import os
        import time
        from pathlib import Path
        now = time.monotonic()
        if not force and now - self.last_time < 2.:
            return
        step = backend.control_step
        elapsed = now - self.last_time
        simulated = 0. if self.last_step is None else (step - self.last_step) * backend.env.step_dt
        self.rss_mb = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/2**20
        self.stream.write(json.dumps(dict(wall_time=time.time(), elapsed_seconds=now-self.started,
            stage=backend.stage, control_step=step, interval_wall_seconds=elapsed,
            interval_sim_seconds=simulated, real_time_factor=simulated/max(elapsed,1e-9),
            rss_mb=self.rss_mb))+'\n')
        self.last_time, self.last_step = now, step

    def close(self):
        self.stream.close()
