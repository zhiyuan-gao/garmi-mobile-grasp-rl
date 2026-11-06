"""Deterministic candidate and recovery rules for the deployed bowl skill."""
import math


def choose_reference(description, attempted, preferred=None):
    if not description['eligible']:
        raise ValueError(description['rejection'])
    allowed = [name for name in description['allowed_references'] if name not in attempted]
    if preferred is not None and not attempted:
        if preferred not in allowed:
            raise ValueError('Requested reference is not allowed at the measured object pose')
        return preferred
    if not allowed:
        raise ValueError('No untried allowed grasp reference remains')
    return allowed[0]


def station_candidates(description, preferred=None, excluded=(), margin=.05):
    """Sample the trained region, prioritizing lateral alignment then travel.

    The margin reserves room for navigation settling. This is a geometric
    heuristic, not a learned success score; CRAM/Giskard must probe each path.
    """
    lower, upper = description['skill']['base_bounds_map_xy']
    yaw = description['skill']['base_yaw']
    low = [x+margin for x in lower]; high = [x-margin for x in upper]
    if any(a>b for a,b in zip(low,high)):
        raise ValueError('Training region is too narrow for the navigation margin')
    current = description['snapshot']['base_map_position']
    target = description['snapshot']['object_map_position']
    x = min(high[0],max(low[0],target[0]))
    xs = [x, (low[0]+high[0])/2, low[0], high[0]]
    ys = [low[1], (low[1]+high[1])/2, high[1]]
    candidates = sorted({(a,b,yaw) for a in xs for b in ys},
        key=lambda p:(abs(p[0]-target[0]), math.dist(p[:2],current[:2])))
    if preferred is not None:
        if (len(preferred)!=3 or not all(math.isfinite(v) for v in preferred)
                or any(not lo <= v <= hi for v,lo,hi in zip(preferred[:2],lower,upper))
                or abs(math.atan2(math.sin(preferred[2]-yaw),math.cos(preferred[2]-yaw)))>.01):
            raise ValueError('Requested navigation pose is outside the skill region or heading')
        candidates.insert(0,tuple(preferred))
    result=[]
    for pose in candidates:
        if pose not in result and not any(math.dist(pose[:2],old[:2])<.02 for old in excluded):
            result.append(pose)
    return result


def recovery_rejection(state, description):
    """Return a reason to stop, or None for a stationary, empty-handed miss."""
    result=state['result']
    if state['owner'] != 'holding' or not result or result['success']:
        return 'Recovery requires a completed failed attempt held by the skill bridge'
    if result['reason'] not in ('attempt_failed','timeout'):
        return f"Failure {result['reason']} requires inspection rather than automatic retry"
    if not description['eligible']:
        return description['rejection'] or 'Object is outside skill applicability'
    snapshot=description['snapshot']; measured=snapshot['grasp_measurements']
    if not measured or not measured['supported'] or measured['contact'] or measured['collision']:
        return 'Recovery requires a supported object, empty gripper and no forbidden collision'
    velocity=snapshot['object_velocity']
    if (len(velocity)!=6 or not all(math.isfinite(v) for v in velocity)
            or math.sqrt(sum(v*v for v in velocity[:3]))>.02
            or math.sqrt(sum(v*v for v in velocity[3:]))>.05):
        return 'Wait for the object to settle before recovery'
    return None
