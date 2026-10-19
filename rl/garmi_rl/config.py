"""Isaac Lab manager-based GARMI grasp task configuration."""
from pathlib import Path
import isaaclab.sim as sim
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import (ObservationGroupCfg, ObservationTermCfg,
    RewardTermCfg, TerminationTermCfg, EventTermCfg)
from isaaclab.utils import configclass
from .actions import PlanarVelocityActionCfg
from . import mdp

ASSETS = Path(__file__).resolve().parents[1] / "assets"
PARK = [0., -.785398, 0., -2.356194, 0., 1.570796, .785398]
BODIES = ["base_link"] + [f"right_fr3_link{i}" for i in range(0,8)] + [
    "right_fr3_leftfinger", "right_fr3_rightfinger"]


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim.UsdFileCfg(usd_path=str(ASSETS/"garmi.usd"),activate_contact_sensors=True,
            rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.),
            articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=2)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,.0259),
            joint_pos={**{f"right_fr3_joint{i+1}":q for i,q in enumerate(PARK)},
                       "right_fr3_finger_joint.*": .04}),
        actuators={
            "arm":ImplicitActuatorCfg(joint_names_expr=["right_fr3_joint[1-7]"],
                effort_limit_sim={"right_fr3_joint[1-4]":87.,"right_fr3_joint[5-7]":12.},
                velocity_limit_sim=1.,stiffness=10000.,damping=200.),
            "fingers":ImplicitActuatorCfg(joint_names_expr=["right_fr3_finger_joint.*"],
                effort_limit_sim=40.,velocity_limit_sim=.04,stiffness=2000.,damping=100.)},
        soft_joint_pos_limit_factor=.95)
    object = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim.UsdFileCfg(usd_path=str(ASSETS/"bowl.usda"),activate_contact_sensors=True,
            rigid_props=sim.RigidBodyPropertiesCfg(solver_position_iteration_count=8,
                solver_velocity_iteration_count=2, max_depenetration_velocity=1.)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(.75,-.25,.9775),rot=(.70710678,0,0,-.70710678)))
    table = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim.CuboidCfg(size=(.6,.8,.04),collision_props=sim.CollisionPropertiesCfg(
            contact_offset=.001,rest_offset=0.),
            rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim.MassPropertiesCfg(mass=100.),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.45,.3,.18))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(.99,-.25,.925)))
    ground = AssetBaseCfg(prim_path="/World/Ground",spawn=sim.GroundPlaneCfg(
        physics_material=sim.RigidBodyMaterialCfg(static_friction=0.,dynamic_friction=0.,
            friction_combine_mode="min")))
    light = AssetBaseCfg(prim_path="/World/Light",spawn=sim.DomeLightCfg(intensity=1800.))
    finger_1 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_fr3_leftfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],history_length=10,update_period=0.)
    finger_2 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_fr3_rightfinger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],history_length=10,update_period=0.)
    support = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Object",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],history_length=1,update_period=0.)


@configclass
class ActionsCfg:
    base = PlanarVelocityActionCfg(asset_name="robot")
    arm = lab_mdp.JointPositionActionCfg(asset_name="robot",joint_names=["right_fr3_joint[1-7]"],
        scale=1.5,use_default_offset=True,clip={
            "right_fr3_joint[13]":(-2.75,2.75),"right_fr3_joint2":(-1.74,1.74),
            "right_fr3_joint4":(-3.,-.19),"right_fr3_joint5":(-2.73,2.73),
            "right_fr3_joint6":(.55,4.51),"right_fr3_joint7":(-2.89,2.89)})
    gripper = lab_mdp.BinaryJointPositionActionCfg(asset_name="robot",
        joint_names=["right_fr3_finger_joint.*"],open_command_expr={".*":.04},close_command_expr={".*":0.})


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        state = ObservationTermCfg(func=mdp.observation)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False
    policy: Policy = Policy()


@configclass
class RewardsCfg:
    approach = RewardTermCfg(func=mdp.approach,weight=1.)
    lift = RewardTermCfg(func=mdp.lift,weight=15.)
    hold = RewardTermCfg(func=mdp.hold,weight=5.)
    action_rate = RewardTermCfg(func=lab_mdp.action_rate_l2,weight=-1.e-4)


@configclass
class GraspRewardsCfg:
    approach = RewardTermCfg(func=mdp.approach,weight=1.)
    close = RewardTermCfg(func=mdp.close_aligned,weight=1.)
    grasp = RewardTermCfg(func=mdp.grasp_contact,weight=2.)
    lift = RewardTermCfg(func=mdp.lift,weight=15.)
    hold = RewardTermCfg(func=mdp.hold,weight=5.)
    action_rate = RewardTermCfg(func=lab_mdp.action_rate_l2,weight=-1.e-4)
    success = RewardTermCfg(func=mdp.completed,weight=0.)


@configclass
class FixedFinishRewardsCfg:
    approach = RewardTermCfg(func=mdp.policy_approach,weight=1.)
    action_rate = RewardTermCfg(func=mdp.policy_action_rate,weight=-1.e-4)
    success = RewardTermCfg(func=mdp.completed,weight=0.)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=lab_mdp.time_out,time_out=True)
    collision = TerminationTermCfg(func=mdp.termination,params={"key":"collision"})
    dropped = TerminationTermCfg(func=mdp.termination,params={"key":"dropped"})
    out_of_bounds = TerminationTermCfg(func=mdp.termination,params={"key":"out_of_bounds"})
    success = TerminationTermCfg(func=mdp.termination,params={"key":"success"})


@configclass
class FixedFinishTerminationsCfg(TerminationsCfg):
    attempt_failed = TerminationTermCfg(func=mdp.termination,params={"key":"attempt_failed"})


@configclass
class EventsCfg:
    reset = EventTermCfg(func=mdp.reset,mode="reset")


@configclass
class GarmiEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1,env_spacing=3.,replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation=10
        self.episode_length_s=15.
        self.sim.dt=.005
        self.sim.render_interval=10
        self.sim.physx.enable_stabilization=True
        self.viewer.eye=(2.5,-3.,2.)
        self.viewer.lookat=(.3,0.,.9)
        self.viewer.resolution=(960,640)
        self.forbidden_sensors=[]
        for i,body in enumerate(BODIES):
            # Root contains the fixed left arm, head, lift and wheeled chassis.
            # Ground support is allowed; fingers may contact the target bowl.
            targets=["{ENV_REGEX_NS}/Table"]
            targets += ["{ENV_REGEX_NS}/Robot/"+b for b in BODIES if b != body]
            if "finger" not in body:
                targets += ["{ENV_REGEX_NS}/Object"]
            if body != "base_link":
                targets += ["/World/Ground/GroundPlane/CollisionPlane"]
            key=f"forbidden_{i}"
            setattr(self.scene,key,ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/"+body,
                filter_prim_paths_expr=targets,history_length=10,update_period=0.))
            self.forbidden_sensors.append(key)


def configured_env(agent_cfg, num_envs, seed):
    """Select a frozen trial variant; legacy PPO files keep the original task."""
    cfg = GarmiEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    task = agent_cfg.get("garmi_task", {"variant": "baseline"})
    cfg.episode_length_s = float(task.get("episode_length_s", cfg.episode_length_s))
    if "observation_time_scale_s" in task:
        cfg.observation_time_scale_s = float(task["observation_time_scale_s"])
        if cfg.observation_time_scale_s <= 0:
            raise ValueError("Observation time scale must be positive")
    variant = task["variant"]
    description = dict(variant=variant)
    if variant == "grasp_v2":
        from .grasp_rewards import describe, WEIGHTS
        cfg.rewards = GraspRewardsCfg()
        description = describe(cfg.sim.dt * cfg.decimation,
                               agent_cfg["agent"]["discount_factor"], task["success_bonus"])
        for name, weight in WEIGHTS.items():
            getattr(cfg.rewards, name).weight = weight
        # RewardManager multiplies ALL terms by dt; completion is one payment.
        cfg.rewards.success.weight = description["success_bonus"] / description["step_dt"]
    elif variant == "fixed_finish_v3":
        from .fixed_finish import (DEFAULTS, PHASES, FinishBaseActionCfg,
                                  FinishArmActionCfg, FinishGripperActionCfg)
        cfg.fixed_finish = {**DEFAULTS, **task.get("controller", {})}
        if set(cfg.fixed_finish) != set(DEFAULTS):
            raise ValueError("Unknown fixed-finish controller setting")
        cfg.actions.base = FinishBaseActionCfg(asset_name="robot")
        previous_arm = cfg.actions.arm
        cfg.actions.arm = FinishArmActionCfg(asset_name="robot", joint_names=previous_arm.joint_names,
            scale=previous_arm.scale, use_default_offset=True, clip=previous_arm.clip)
        cfg.actions.gripper = FinishGripperActionCfg(asset_name="robot")
        cfg.rewards = FixedFinishRewardsCfg()
        cfg.terminations = FixedFinishTerminationsCfg()
        dt = cfg.sim.dt * cfg.decimation
        bonus = float(task["success_bonus"])
        cfg.rewards.success.weight = bonus / dt
        duration = sum(cfg.fixed_finish[k] for k in (
            "approach_seconds", "settle_seconds", "close_seconds", "lift_seconds", "hold_seconds"))
        if duration + dt >= cfg.episode_length_s:
            raise ValueError("The full fixed attempt must fit inside the episode limit")
        description = dict(variant=variant, controller=cfg.fixed_finish, phases=PHASES,
            policy_action_dim=10, policy_observation_dim=64, step_dt=dt,
            weights=dict(approach=1., action_rate=-1.e-4), success_bonus=bonus,
            success_bonus_source="Retained from trial-02; paid once only on physical success",
            shaping_only_during_approach=True, gripper_policy_control=False,
            failure_is_terminal=True, terminal_failure_reward=0.,
            method="PPO with a deterministic finishing controller; not a GA-DDPG reproduction")
        approach_reward = task.get("approach_reward", "legacy")
        if approach_reward == "gap_pose_v1":
            from . import gap_rewards
            cfg.rewards.approach.func = gap_rewards.policy_approach
            description["approach_reward"] = gap_rewards.description()
            parameters = task.get("approach_reward_parameters", {})
            if set(parameters) - {"distance_clip_m", "distance_scale_per_m"}:
                raise ValueError("Unknown GAP pose reward parameter")
            if any(float(v) <= 0 for v in parameters.values()):
                raise ValueError("Distance reward parameters must be positive")
            if parameters:
                cfg.rewards.approach.params = dict(parameters)
                clip = parameters.get("distance_clip_m", .5)
                scale = parameters.get("distance_scale_per_m", 5.)
                description["approach_reward"].update(parameters=parameters,
                    formula=f"0.5*(1-tanh({scale}*clip(d,0,{clip}))) + 0.5*(1-tanh(5*clip(1-cos(theta/2),0,0.5)))")
                description["approach_reward"]["adaptations"].append(
                    "Kitchen distance range adaptation; original scale/clipping retained by default")
        elif approach_reward != "legacy":
            raise ValueError(f"Unknown approach reward: {approach_reward}")
    elif variant != "baseline":
        raise ValueError(f"Unknown GARMI task variant: {variant}")
    if 'action_space' in task:
        from .joint_delta import ACTION_SPACE, JOINT_SCALE
        if task['action_space'] != ACTION_SPACE or variant != 'fixed_finish_v3':
            raise ValueError('Unknown action space or incompatible task variant')
        from .joint_delta_action import JointDeltaArmActionCfg
        previous_arm = cfg.actions.arm
        cfg.actions.arm = JointDeltaArmActionCfg(asset_name='robot', joint_names=previous_arm.joint_names,
            scale=1., use_default_offset=False, clip=previous_arm.clip)
        cfg.joint_delta_actions = True
        description.update(action_space=ACTION_SPACE, policy_action_dim=10,
            joint_increment_scale_rad=JOINT_SCALE,
            joint_target='current measured angle + clipped action * .05 rad',
            observation_history='previous three base commands and seven joint increments',
            approach_inverse_kinematics=False)
    if "multi_object" in task:
        from .multi_object import configure
        description["multi_object"] = configure(cfg, task["multi_object"])
    if "kitchen" in task:
        from .kitchen import configure
        description["kitchen"] = configure(cfg, task["kitchen"])
    description["episode_length_s"] = cfg.episode_length_s
    description["observation_time_scale_s"] = getattr(cfg, "observation_time_scale_s", cfg.episode_length_s)
    return cfg, description
