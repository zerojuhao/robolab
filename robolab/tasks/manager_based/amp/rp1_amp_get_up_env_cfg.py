# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from dataclasses import MISSING

from isaaclab.envs.mdp import base_pos_z
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robolab.tasks.manager_based.amp.mdp as mdp
from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import RP1_3_CFG
from robolab.tasks.manager_based.amp.managers import AnimationTermCfg as AnimTerm
from robolab.tasks.manager_based.amp.managers import MotionDataTermCfg as MotionDataTerm
from robolab.tasks.manager_based.amp.amp_env_cfg import AmpEnvCfg

KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
]
AMP_NUM_STEPS = 8
BASE_BODY_NAME = "waist_yaw_link"
INITIAL_ASSIST_FORCE = 200.0
MOTION_JOINT_POS_NOISE = 0.1
FALLEN_JOINT_POS_NOISE = 0.3
MAX_MOTION_PHASE = 0.3
TARGET_BASE_HEIGHT_PHASE3 = 0.55
BASE_HEIGHT_TARGET = 0.75

MOTION_DATA_WEIGHTS = {
    "fallAndGetUp1_subject1_1060_1150": 1.0,
    "fallAndGetUp1_subject1_1400_1480": 1.0,
    "fallAndGetUp1_subject1_2100_2200": 1.0,
    "fallAndGetUp1_subject5_2500_2600": 1.0,
    "fallAndGetUp2_subject2_850_1050": 1.0,
    "fallAndGetUp2_subject3_900_1000": 1.0,
    "fallAndGetUp1_subject5_2100_2200": 1.0,
    "fallAndGetUp1_subject5_3900_3980": 1.0,
    "fallAndGetUp2_subject3_450_550": 1.0,
    "fallAndGetUp1_subject1_680_800": 1.0,
    "fallAndGetUp1_subject1_850_940": 1.0,
    "fallAndGetUp1_subject1_1600_1700": 1.0,
    "fallAndGetUp1_subject1_2300_2400": 1.0,
    "fallAndGetUp1_subject4_3700_3800": 1.0,
    "fallAndGetUp2_subject2_360_580": 1.0,
    "fallAndGetUp2_subject2_1200_1370": 1.0,
    "fallAndGetUp2_subject2_1500_1600": 1.0,
    "fallAndGetUp2_subject3_1850_1920": 1.0,
    "fallAndGetUp2_subject3_2080_2180": 1.0,
}

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

@configclass
class CommandsCfg:
    """Velocity command aligned with locomotion AMP for downstream task fusion."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ObservationsCfg():
        
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.35, n_max=0.35))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.03, n_max=0.03))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.75, n_max=1.75))
        actions = ObsTerm(func=mdp.last_action)
        

        def __post_init__(self):
            self.history_length = 8
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group. (has privilege observations)"""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=KEY_BODY_NAMES,
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self):
            self.history_length = 8
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class DiscriminatorCfg(ObsGroup):
        ref_root_projected_gravity = ObsTerm(
            func=mdp.ref_root_projected_gravity,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            },
        )
        root_local_rot_tan_norm = ObsTerm(func=mdp.root_local_rot_tan_norm)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        key_body_pos_b = ObsTerm(
            func=mdp.key_body_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=KEY_BODY_NAMES,
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1
            self.history_length = 8
            self.flatten_history_dim = False

    disc: DiscriminatorCfg = DiscriminatorCfg()

    @configclass
    class DiscriminatorDemoCfg(ObsGroup):
        ref_root_projected_gravity = ObsTerm(
            func=mdp.ref_root_projected_gravity,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            },
        )
        ref_root_local_rot_tan_norm = ObsTerm(
            func=mdp.ref_root_local_rot_tan_norm,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            },
        )
        ref_root_ang_vel_b = ObsTerm(
            func=mdp.ref_root_ang_vel_b,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            },
        )
        ref_joint_pos = ObsTerm(
            func=mdp.ref_joint_pos,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            },
        )
        ref_joint_vel = ObsTerm(
            func=mdp.ref_joint_vel,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            },
        )
        ref_key_body_pos_b = ObsTerm(
            func=mdp.ref_key_body_pos_b,
            params={
                "animation": "animation",
                "flatten_steps_dim": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.concatenate_dim = -1

    disc_demo: DiscriminatorDemoCfg = DiscriminatorDemoCfg()


@configclass
class RewardsCfg:
    # Base
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    
    # standing
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy,
        weight=2.0,
        params={
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    lin_vel_xy = RewTerm(
        func=mdp.lin_vel_xy,
        weight=2.0,
        params={
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    target_orientation = RewTerm(
        func=mdp.target_orientation,
        weight=2.0,
        params={
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    target_base_height = RewTerm(
        func=mdp.target_base_height,
        weight=5.0,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    target_joint_deviation_l2 = RewTerm(
        func=mdp.target_joint_deviation_l2,
        weight=-0.1,
        params={
            "target_base_height_phase3": TARGET_BASE_HEIGHT_PHASE3,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class EventsCfg:
    """Configuration for events."""
    
    # startup
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "base_link"]),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.055, 0.055), "z": (-0.055, 0.055)},
        },
    )

    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_.*_link", "right_.*_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_joint"]),
            "friction_distribution_params": (1.0, 1.0),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    
    # interval
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": VELOCITY_RANGE},
    )
    
    init_assist_force = EventTerm(
        func=mdp.init_assist_force,
        mode="startup",
        params={
            "initial_force": INITIAL_ASSIST_FORCE,
        },
    )

    reset_get_up_robust = EventTerm(
        func=mdp.reset_get_up_robust,
        mode="reset",
        params={
            "motion_data_term": "motion_dataset",
            "asset_cfg": SceneEntityCfg("robot"),
            "motion_fallen_prob": 0.6,
            "fallen_deviated_prob": 0.4,
            "max_motion_phase": MAX_MOTION_PHASE,
            "height_offset": 0.2,
            "yaw_range": (-math.pi, math.pi),
            "yaw_noise": 0.35,
            "motion_joint_pos_noise": MOTION_JOINT_POS_NOISE,
            "fallen_joint_pos_noise": FALLEN_JOINT_POS_NOISE,
            "fallen_root_height_range": (0.2, 0.4),
        },
    )

    apply_force = EventTerm(
        func=mdp.apply_force,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_BODY_NAME]),
            "initial_force": INITIAL_ASSIST_FORCE,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    force_level = CurrTerm(
        func=mdp.force_level,
        params={
            "reward_term_name": "target_base_height",
            "initial_force": INITIAL_ASSIST_FORCE,
            "force_decrement": 10.0,
        },
    )

@configclass
class MotionDataCfg:
    """Motion data settings for the MDP."""
    motion_dataset = MotionDataTerm(
        motion_data_dir="", 
        motion_data_weights={},
    )
    
@configclass
class AnimationCfg:
    """Animation settings for the MDP."""
    animation = AnimTerm(
        motion_data_term="motion_dataset",
        motion_data_components=[
            "root_pos_w",
            "root_quat",
            "root_vel_w",
            "root_ang_vel_w",
            "dof_pos",
            "dof_vel",
            "key_body_pos_b",
        ], 
        num_steps_to_use=10, 
        random_initialize=True,
        random_fetch=True,
        enable_visualization=False,
    )
    
@configclass
class RP1AmpGetUpEnvCfg(AmpEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    motion_data: MotionDataCfg = MotionDataCfg()
    animation: AnimationCfg = AnimationCfg()
    
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.robot = RP1_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rp1_24dof_getup_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights = MOTION_DATA_WEIGHTS

        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        self.events.add_base_mass.params["asset_cfg"].body_names = BASE_BODY_NAME


@configclass
class RP1AmpGetUpEnvCfg_PLAY(RP1AmpGetUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.events.init_assist_force.params["initial_force"] = 0.0
        self.events.apply_force.params["initial_force"] = 0.0

        self.observations.policy.enable_corruption = False
        self.animation.animation.random_initialize = False
        self.animation.animation.random_fetch = False
