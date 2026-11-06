"""Read-only skill metadata and measured applicability, evaluated in the simulator."""
import math

from garmi_cram_rl.contract import object_candidates, validate_start


def describe_skill(backend):
    snapshot=backend.snapshot()
    diagnostics=backend.mdp.diagnostics(backend.env)
    snapshot['grasp_measurements']={name:float(diagnostics[name][0]) for name in
        ('height','slip','contact','supported','collision')}
    skill=dict(object_name=backend.mask.data['object_name'], arm='RIGHT',
        checkpoint_sha256=backend.bundle.manifest['checkpoint_sha256'],
        action_contract=backend.bundle.manifest['action_contract'],
        base_bounds_map_xy=backend.task['kitchen']['base_bounds_map_xy'],
        base_yaw=math.radians(backend.task['kitchen']['base_map_yaw_degrees']),
        object_regions=[r['bounds_map_xy'] for r in backend.mask.data['regions']],
        object_origin_z=backend.mask.data['bowl_origin_z_m'],
        reference_ids=list(backend.references))
    rejection=None; allowed=[]; admission=None
    try:
        allowed=object_candidates(snapshot,backend.mask)
        if not snapshot['grasp_measurements']['supported']:
            raise ValueError('Object is not supported on the counter')
        if snapshot['grasp_measurements']['collision']:
            raise ValueError('Robot is in forbidden contact')
        velocity=snapshot['object_velocity']
        if (not all(math.isfinite(v) for v in velocity)
                or math.sqrt(sum(v*v for v in velocity[:3]))>.02
                or math.sqrt(sum(v*v for v in velocity[3:]))>.05):
            raise ValueError('Object is not stationary on the counter')
    except ValueError as error:
        rejection=str(error)
    if rejection is None:
        try:
            validate_start(snapshot,backend.task,backend.mask)
        except ValueError as error:
            admission=str(error)
    return dict(skill=skill, snapshot=snapshot, allowed_references=allowed,
                eligible=rejection is None, rejection=rejection, admission=admission)
