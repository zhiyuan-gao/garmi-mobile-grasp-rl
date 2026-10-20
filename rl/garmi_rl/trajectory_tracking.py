"""Planar trajectory tracking in the base-link frame, without changing the path."""
import numpy as np
import copy


def wrap_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def base_tracking_velocity(previous, following, measured, dt, response_time):
    """World-frame feedforward plus position feedback, returned in measured base axes.

    The caller enforces the existing speed and acceleration limits. Once the
    trajectory ends, pass the final pose as both previous and following.
    """
    previous, following, measured = (np.asarray(x, dtype=float) for x in (previous, following, measured))
    if dt <= 0 or response_time <= 0:
        raise ValueError('Tracking periods must be positive')
    delta, error = following-previous, previous-measured
    delta[2], error[2] = wrap_angle(delta[2]), wrap_angle(error[2])
    velocity = delta/dt+error/response_time
    c, s = np.cos(measured[2]), np.sin(measured[2])
    return np.array([c*velocity[0]+s*velocity[1], -s*velocity[0]+c*velocity[1], velocity[2]])


def retime_plan(plan, dt, speed_limits, acceleration_limits):
    """Slow a rest-start path until sampled base commands fit limits.

    Original waypoints remain exact; no new route or grasp pose is introduced.
    Include startup and final stop in the acceleration audit. The caller keeps
    the original episode deadline even if this trajectory becomes too long.
    """
    if not np.allclose(plan['initial'].get('base_velocity', [0,0,0]), 0., atol=1.e-8):
        raise ValueError('This reset-state retimer requires a stationary initial base')
    states = [plan['initial']]+plan['trajectory']
    poses = np.array([x['base_pose'] for x in states])
    poses[:,2] = np.unwrap(poses[:,2])
    speed_limits, acceleration_limits = np.asarray(speed_limits), np.asarray(acceleration_limits)
    if dt <= 0 or np.any(speed_limits <= 0) or np.any(acceleration_limits <= 0):
        raise ValueError('Limits must be positive')
    for factor in range(1, 101):
        samples = np.array([poses[k]+(poses[k+1]-poses[k])*j/factor
                            for k in range(len(poses)-1) for j in range(1,factor+1)])
        delta = np.diff(np.vstack([poses[0],samples]),axis=0)/dt
        c,s = np.cos(samples[:,2]),np.sin(samples[:,2])
        velocities = np.column_stack([c*delta[:,0]+s*delta[:,1], -s*delta[:,0]+c*delta[:,1],delta[:,2]])
        accelerations = np.diff(np.vstack([np.zeros(3),velocities,np.zeros(3)]),axis=0)/dt
        if np.all(np.abs(velocities)<=speed_limits+1.e-9) and np.all(np.abs(accelerations)<=acceleration_limits+1.e-9):
            break
    else:
        raise ValueError('Could not retime path within 100x duration')
    result = copy.deepcopy(plan)
    result['trajectory'] = []
    for index, pose in enumerate(samples):
        segment, substep = divmod(index, factor)
        fraction = (substep+1)/factor
        a,b = states[segment],states[segment+1]
        record = copy.deepcopy(b)
        record['base_pose'] = pose.tolist()
        record['base_velocity'] = velocities[index].tolist()
        record['arm'] = (np.asarray(a['arm'])*(1-fraction)+np.asarray(b['arm'])*fraction).tolist()
        # This interpolated hand position is only an approximate tracking metric.
        # Execution uses the interpolated joint and base positions above.
        hand = np.array(b['hand'])
        hand[:3,3] = np.array(a['hand'])[:3,3]*(1-fraction)+hand[:3,3]*fraction
        record['hand'] = hand.tolist()
        result['trajectory'].append(record)
    result['ticks'] = len(samples)
    result['retiming'] = dict(factor=factor, seconds=len(samples)*dt,
        peak_base_velocity=np.max(np.abs(velocities),axis=0).tolist(),
        peak_base_acceleration=np.max(np.abs(accelerations),axis=0).tolist())
    return result
