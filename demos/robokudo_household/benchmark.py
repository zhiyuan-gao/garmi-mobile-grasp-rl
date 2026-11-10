"""Compare original/contact-fast/on-demand-camera costs in the same scene."""
import argparse
from pathlib import Path
import sys
import json
import time
import os
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'household_rl'),str(Path(__file__).parent),str(ROOT/'demos/household_sequence'),str(ROOT/'demos/robokudo_bowl'),str(ROOT/'demos/mug_drawer'),str(ROOT/'integration'),str(ROOT/'rl'),str(ROOT/'cram-vrb-lab')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=True;a.out.mkdir(parents=True,exist_ok=False)
app=AppLauncher(a,renderer='RaytracedLighting',width=640,height=360).app
from garmi_cram_rl.contract import Bundle
from household_perception import PerceivedHouseholdBackend
from performance import optimize_contact_updates,restore_contact_updates
import torch
backend=None

def rss():
    return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/2**20

try:
    backend=PerceivedHouseholdBackend(Bundle(ROOT/'results/cram-rl-integration/policy-7500',ROOT),
        ROOT/'household/assets',ROOT/'household/mug',a.out,render=False,display=True)
    app.set_setting('/app/runLoops/main/rateLimitEnabled',False)
    # Compare the exact original and optimized update on the same PhysX state.
    # Include a one-substep contact pulse to verify history and threshold inputs.
    sensor=backend.env.scene['forbidden_0']
    original=sensor._original_contact_update
    original_matrix=sensor.contact_physx_view.get_contact_force_matrix
    physical=original_matrix(dt=sensor._sim_physics_dt).clone()
    def data():
        return {k:v.clone() for k,v in vars(sensor._data).items() if isinstance(v,torch.Tensor)}
    def save():
        return data(),sensor._timestamp.clone(),sensor._timestamp_last_update.clone(),sensor._is_outdated.clone()
    def restore(state):
        for k,v in state[0].items():getattr(sensor._data,k).copy_(v)
        sensor._timestamp.copy_(state[1]);sensor._timestamp_last_update.copy_(state[2]);sensor._is_outdated.copy_(state[3])
    checks=[]
    for i in range(14):
        value=torch.zeros_like(physical)
        if i==2:value.reshape(-1)[0]=17.
        sensor.contact_physx_view.get_contact_force_matrix=lambda dt,value=value:value
        before=save()
        # Invoke the original acquisition path explicitly, even though methods
        # on this instance have been optimized.
        sensor._timestamp.add_(backend.env.physics_dt);sensor._is_outdated.fill_(True)
        original['buffers']();expected=save()
        restore(before);sensor.update(backend.env.physics_dt)
        actual=save()
        for k,v in expected[0].items():torch.testing.assert_close(actual[0][k],v,rtol=0,atol=0,equal_nan=True)
        for v,w in zip(actual[1:],expected[1:]):torch.testing.assert_close(v,w,rtol=0,atol=0)
        checks.append(float(sensor.data.force_matrix_w_history.abs().max()))
    sensor.contact_physx_view.get_contact_force_matrix=original_matrix
    assert checks[2:12]==[17.]*10 and checks[12:]==[0.,0.],checks
    results={'contact_history_exact_match':True,'pulse_history_max':checks,'cases':[]}
    for label,fast,camera_on in [('original',False,True),('fast_contacts',True,True),('fast_contacts_camera_idle',True,False)]:
        (optimize_contact_updates if fast else restore_contact_updates)(backend.env.scene)
        camera=backend.sensor.camera
        camera._render_product.hydra_texture.set_updates_enabled(camera_on)
        if camera_on and camera.is_paused():camera.resume()
        elif not camera_on:camera.pause()
        for _ in range(10):backend.step_classical(time.monotonic())
        times=[];start_rss=rss()
        for _ in range(100):
            start=time.perf_counter();backend.step_classical(time.monotonic());times.append(time.perf_counter()-start)
        row=dict(case=label,steps=len(times),step_ms=1000*sum(times)/len(times),simulated_seconds=len(times)*backend.env.step_dt,
            wall_seconds=sum(times),rss_start_mb=start_rss,rss_end_mb=rss())
        results['cases'].append(row);print('BENCHMARK',json.dumps(row),flush=True)
    (a.out/'benchmark.json').write_text(json.dumps(results,indent=2))
finally:
    if backend:backend.close()
    app.close(skip_cleanup=True)
