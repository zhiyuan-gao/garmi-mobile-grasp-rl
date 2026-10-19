"""Task terms. Cached diagnostics are updated once per policy step."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import (
    combine_frame_transforms, subtract_frame_transforms, quat_apply_inverse,
    quat_from_euler_xyz, quat_error_magnitude,
)

ROOT = Path(__file__).resolve().parents[2]
CONTACT_THRESHOLD = .1  # Newton; detection floor, not a severity classification.


def initialize(env):
    if hasattr(env, "grasp_state"):
        return
    n, d = env.num_envs, env.device
    selected = json.loads((ROOT / "results/mobile-grasp/selected.json").read_text())
    cache = np.load(ROOT / "results/mobile-grasp/bowl-grasps.npz",allow_pickle=False)
    mesh = ROOT / "cram-vrb-lab/cognitive_robot_abstract_machine/coraplex/resources/objects/bowl.stl"
    if str(cache["mesh_sha256"]) != hashlib.sha256(mesh.read_bytes()).hexdigest():
        raise ValueError("Cached GraspGen prior does not match the bowl geometry")
    if not np.any(np.all(np.isclose(cache["grasps"],selected["mesh_T_gripper"],atol=1e-6),axis=(1,2))):
        raise ValueError("Selected grasp is not a member of the validated cache")
    transform = np.array(selected["mesh_T_gripper"])
    transform[:3, :3] = transform[:3, :3] @ Rotation.from_euler("z", -np.pi/2).as_matrix()
    quat = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3,0,1,2]]
    env.grasp_pos_o = torch.tensor(transform[:3,3], dtype=torch.float32, device=d).repeat(n,1)
    env.grasp_quat_o = torch.tensor(quat, dtype=torch.float32, device=d).repeat(n,1)
    env.hand_id = env.scene["robot"].find_bodies("right_fr3_link7")[0][0]
    env.observed_joint_ids = env.scene["robot"].find_joints(
        ["right_fr3_joint[1-7]", "right_fr3_finger_joint.*"])[0]
    env.finger_joint_ids = env.scene["robot"].find_joints("right_fr3_finger_joint.*")[0]
    # The hand is merged into link7. Read its exact fixed-chain transform from
    # the asset builder rather than assuming link7 and hand are identical.
    meta = json.loads((ROOT / "rl/assets/manifest.json").read_text())
    t = np.array(meta["link7_T_hand"])
    hq = Rotation.from_matrix(t[:3,:3]).as_quat()[[3,0,1,2]]
    env.hand_offset_p = torch.tensor(t[:3,3], device=d, dtype=torch.float32).repeat(n,1)
    env.hand_offset_q = torch.tensor(hq, device=d, dtype=torch.float32).repeat(n,1)
    env.grasp_state = {
        "hold": torch.zeros(n, device=d), "ever_lifted": torch.zeros(n,device=d,dtype=torch.bool),
        "anchor": torch.zeros(n,3,device=d), "held_last": torch.zeros(n,device=d,dtype=torch.bool),
        "initial_z": torch.full((n,), .9775, device=d),
    }
    env.diagnostic_step = -1
    if hasattr(env.cfg, "multi_object"):
        from .multi_object import GraspBank
        env.grasp_bank = GraspBank(env)


def geometry(env):
    initialize(env)
    r, o = env.scene["robot"], env.scene["object"]
    hp, hq = combine_frame_transforms(r.data.body_pos_w[:,env.hand_id],
        r.data.body_quat_w[:,env.hand_id], env.hand_offset_p, env.hand_offset_q)
    gp, gq = combine_frame_transforms(o.data.root_pos_w, o.data.root_quat_w,
        env.grasp_pos_o, env.grasp_quat_o)
    return hp, hq, gp, gq


def diagnostics(env):
    initialize(env)
    if env.diagnostic_step == env.common_step_counter:
        return env.diagnostics
    hp, hq, gp, gq = geometry(env)
    obj = env.scene["object"].data.root_pos_w
    state = env.grasp_state
    fingers = torch.stack([env.scene[f"finger_{i}"].data.force_matrix_w.norm(dim=-1).amax(dim=(1,2))
                           for i in (1,2)], dim=-1)
    contact = (fingers > CONTACT_THRESHOLD).all(dim=-1)
    rel = quat_apply_inverse(hq, obj-hp)
    new = contact & ~state["held_last"]
    state["anchor"][new] = rel[new]
    slip = (rel-state["anchor"]).norm(dim=-1)
    height = obj[:,2]-state["initial_z"]
    supported = env.scene["support"].data.force_matrix_w.norm(dim=-1).amax(dim=(1,2)) > CONTACT_THRESHOLD
    forbidden = torch.stack([env.scene[k].data.force_matrix_w_history.norm(dim=-1).flatten(1).amax(dim=1)
                             for k in env.forbidden_sensors], dim=1)
    collision = forbidden.amax(dim=1) > CONTACT_THRESHOLD
    held = contact & (height >= .08) & ~supported & (slip < .02) & ~collision
    state["hold"] = torch.where(held, state["hold"] + env.step_dt, 0.)
    state["ever_lifted"] |= contact & (height > .04)
    dropped = (obj[:,2] < .85) | (state["ever_lifted"] & ~contact & (height < .02))
    state["held_last"][:] = contact
    localbase = env.scene["robot"].data.root_pos_w-env.scene.env_origins
    out = (localbase[:,0] < -.5) | (localbase[:,0] > .5) | (localbase[:,1].abs() > .65)
    if hasattr(env.cfg, "kitchen"):
        bounds = env.cfg.kitchen["movement_bounds_task_xy"]
        out = ((localbase[:, 0] < bounds[0][0]) | (localbase[:, 0] > bounds[1][0])
               | (localbase[:, 1] < bounds[0][1]) | (localbase[:, 1] > bounds[1][1]))
        dropped = ((obj[:, 2] < env.cfg.kitchen["drop_height_m"])
                   | (state["ever_lifted"] & ~contact & (height < .02)))
    env.diagnostics = dict(finger_force=fingers, contact=contact, height=height, slip=slip,
        held=held, supported=supported, forbidden_force=forbidden, collision=collision,
        dropped=dropped, out_of_bounds=out, success=(state["hold"] >= 2.) & ~collision,
        distance=(gp-hp).norm(dim=-1), angle=quat_error_magnitude(hq,gq),
        base_position=env.scene["robot"].data.root_pos_w.clone(),
        joint_position=env.scene["robot"].data.joint_pos.clone(),
        gripper_width=env.scene["robot"].data.joint_pos[:,env.finger_joint_ids].sum(-1).clone(),
        object_position=obj.clone(),hand_position=hp.clone())
    if hasattr(env, "grasp_bank"):
        env.diagnostics.update(object_id=env.grasp_bank.object_ids.clone(),
                               reference_id=env.grasp_bank.reference_ids.clone())
    if hasattr(env, "fixed_finish"):
        finish = env.fixed_finish
        env.diagnostics.update(phase=finish.phase.clone(), phase_steps=finish.phase_steps.clone(),
            handover_trigger=finish.trigger.clone(),
            handover_distance=finish.handover_distance.clone(),
            handover_angle=finish.handover_angle.clone(),
            attempt_failed=(finish.phase == 4) & (finish.phase_steps >= finish.ticks("hold_seconds"))
                & ~env.diagnostics["success"])
    env.diagnostic_step = env.common_step_counter
    return env.diagnostics


def observation(env):
    hp,hq,gp,gq = geometry(env)
    robot,obj = env.scene["robot"],env.scene["object"]
    relp,relq = subtract_frame_transforms(hp,hq,gp,gq)
    # Canonical quaternion sign avoids q/-q discontinuity in the observation.
    relq = torch.where(relq[:,:1] < 0, -relq, relq)
    basevel = torch.cat((robot.data.root_lin_vel_b[:,:2],robot.data.root_ang_vel_b[:,2:]),dim=-1)
    object_b = quat_apply_inverse(robot.data.root_quat_w,obj.data.root_pos_w-robot.data.root_pos_w)
    fingers = torch.stack([env.scene[f"finger_{i}"].data.force_matrix_w.norm(dim=-1).amax(dim=(1,2))
                           for i in (1,2)],dim=-1)
    ids=env.observed_joint_ids
    previous_action = env.action_manager.action
    if hasattr(env, "fixed_finish"):
        previous_action = previous_action * (env.fixed_finish.phase == 0)[:, None]
    state = torch.cat(((robot.data.joint_pos-robot.data.default_joint_pos)[:,ids],robot.data.joint_vel[:,ids],
        basevel,env.action_manager.get_term("base").velocity,object_b,
        quat_apply_inverse(hq,obj.data.root_pos_w-hp),relp,relq,
        (fingers>CONTACT_THRESHOLD).float(), previous_action),dim=-1)
    if hasattr(env, "fixed_finish"):
        finish = env.fixed_finish
        scripted = (finish.phase > 0)[:, None]
        target_p = torch.where(scripted, finish.target_position, gp)
        target_q = torch.where(scripted, finish.target_rotation, gq)
        delta_p, delta_q = subtract_frame_transforms(hp, hq, target_p, target_q)
        delta_q = torch.where(delta_q[:, :1] < 0, -delta_q, delta_q)
        state = torch.cat((state, torch.nn.functional.one_hot(finish.phase, 5).float(),
            (finish.phase_steps * env.step_dt / getattr(env.cfg, "observation_time_scale_s", env.cfg.episode_length_s))[:, None], delta_p, delta_q,
            (obj.data.root_pos_w[:, 2]-env.grasp_state["initial_z"])[:, None],
            (env.grasp_state["hold"]/2.)[:, None]), dim=-1)
    return state


def reset(env, env_ids):
    initialize(env)
    r,o = env.scene["robot"],env.scene["object"]
    n = len(env_ids)
    root = r.data.default_root_state[env_ids].clone()
    root[:,:3] += env.scene.env_origins[env_ids]
    offsets = torch.rand(n,3,device=env.device)*2-1
    if getattr(env,"deterministic_reset",False):
        offsets.zero_()
    if getattr(env,"reset_corners",False):
        # Validation-only deterministic vertices of the approved spawn box.
        idx = torch.arange(n,device=env.device) % 8
        offsets = torch.stack([((idx >> bit)&1)*2-1 for bit in range(3)],dim=-1)
    root[:,0] += offsets[:,0]*.05
    root[:,1] += offsets[:,1]*.10
    z = torch.zeros(n,device=env.device)
    root[:,3:7] = quat_from_euler_xyz(z,z,offsets[:,2]*(np.pi/18))
    multi_state = None
    if hasattr(env, "grasp_bank"):
        root, multi_state = env.grasp_bank.reset(env_ids)
    r.write_root_pose_to_sim(root[:,:7],env_ids)
    r.write_root_velocity_to_sim(root[:,7:],env_ids)
    q = r.data.default_joint_pos[env_ids].clone()
    r.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)
    r.set_joint_position_target(q,env_ids=env_ids)
    rs = o.data.default_root_state[env_ids].clone()
    rs[:,:3] += env.scene.env_origins[env_ids]
    if multi_state is not None:
        rs = multi_state
    o.write_root_pose_to_sim(rs[:,:7],env_ids)
    o.write_root_velocity_to_sim(rs[:,7:],env_ids)
    for k in ("hold","ever_lifted","held_last","anchor"):
        env.grasp_state[k][env_ids] = 0
    env.grasp_state["initial_z"][env_ids] = rs[:,2]
    if hasattr(env, "fixed_finish"):
        env.fixed_finish.reset(env_ids)
        if multi_state is not None:
            # Initialize controller state consistently with this episode's goal.
            # Approach observations already use gp/gq directly.
            gp,gq=combine_frame_transforms(rs[:,:3],rs[:,3:7],
                env.grasp_pos_o[env_ids],env.grasp_quat_o[env_ids])
            env.fixed_finish.target_position[env_ids]=gp
            env.fixed_finish.target_rotation[env_ids]=gq
    env.diagnostic_step = -1


def termination(env, key):
    return diagnostics(env)[key]


def approach(env):
    d=diagnostics(env)
    return (1.-torch.tanh(d["distance"]/.1)) * (.5+.5*torch.exp(-d["angle"]))


def policy_approach(env):
    return approach(env) * (env.fixed_finish.phase == 0)


def policy_action_rate(env):
    delta = env.action_manager.action-env.action_manager.prev_action
    return delta.square().sum(-1) * (env.fixed_finish.phase == 0)


def lift(env):
    d=diagnostics(env)
    return (d["height"]/.08).clamp(0.,1.) * d["contact"] * ~d["supported"]


def hold(env):
    d=diagnostics(env)
    return d["held"].float()


def close_aligned(env):
    from .grasp_rewards import formation_terms
    d = diagnostics(env)
    return formation_terms(d["distance"], d["angle"], d["gripper_width"],
                           d["contact"], d["collision"])[0]


def grasp_contact(env):
    d = diagnostics(env)
    return (d["contact"] & ~d["collision"]).float()


def completed(env):
    d = diagnostics(env)
    return (d["success"] & ~d["collision"] & ~d["dropped"] & ~d["out_of_bounds"]).float()
