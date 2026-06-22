
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import robolab.tasks.manager_based.perceptive_hoi.mdp as mdp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from robolab.sensors import NoisyGroupedRayCasterCameraCfg
from robolab.utils.noise import (
    CropAndResizeCfg,
    DepthNormalizationCfg,
    DepthSkyArtifactNoiseCfg,
    GaussianBlurNoiseCfg,
    PerlinNoiseCfg,
    PixelFailureNoiseCfg,
    RandomConvNoiseCfg,
    ScaleRandomizationNoiseCfg,
    StereoFusionNoiseCfg,
)

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}
POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
}
OBJECT_POSE_RANGE = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (-0.01, 0.01),
    "roll": (-0.05, 0.05),
    "pitch": (-0.05, 0.05),
    "yaw": (-0.08, 0.08),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the perceptive HOI scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = MISSING
    robot_reference: ArticulationCfg | None = None
    box_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box_object",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.4, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
        ),
    )
    box_object_reference: RigidObjectCfg | None = None
    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        mesh_prim_paths=[
            "/World/ground",
            # NOTE: Don't forget to add the robot links in robot-specific configuration file.
        ],
        ray_alignment="yaw",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),  # fovx
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),  # fovy
            width=64,
            height=36,
        ),
        debug_vis=False,
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        depth_clipping_behavior="max",
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(
                0.0875,
                0.01,
                0.20568,
            ),
            rot=(
                0.866,
                0.0,
                0.5,
                0.0,
            ),
            convention="world",
        ),
        min_distance=0.1,
        noise_pipeline={
            "scale_randomization": ScaleRandomizationNoiseCfg(
                apply_probability=0.5,
                scale_min=0.97,
                scale_max=1.03,
            ),
            "stereo_fusion": StereoFusionNoiseCfg(
                apply_probability=0.4,
                disparity_grad_threshold=0.10,
                texture_var_threshold=3e-4,
                hole_probability=0.02,
                hole_kernel_size=1,
                hole_value=2.5,
            ),
            "random_conv": RandomConvNoiseCfg(
                apply_probability=0.3,
                kernel_std=0.05,
                center_weight=1.0,
            ),
            "perlin_noise": PerlinNoiseCfg(
                apply_probability=0.5,
                octaves=3,
                base_frequency=8.0,
                lacunarity=2.0,
                persistence=0.5,
                amplitude=1.0,
                noise_std=0.01,
            ),
            "pixel_failures": PixelFailureNoiseCfg(
                apply_probability=0.5,
                dead_pixel_prob=5e-4,
                saturated_pixel_prob=5e-4,
                dead_value=0.0,
                saturated_value=2.5,
            ),
            "sky_artifact_noise": DepthSkyArtifactNoiseCfg(),
            "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
            "depth_normalization": DepthNormalizationCfg(
                depth_range=(0.0, 2.5),
                normalize=True,
                output_range=(0.0, 1.0),
            ),
        },
        data_histories={"distance_to_image_plane_noised": 37},
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range=POSE_RANGE,
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        object_pose_range=OBJECT_POSE_RANGE,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
            history_length=8,
            flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=8,
            flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=8,
            flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            history_length=8,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=8,
            flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=8, flatten_history_dim=True, clip=(-10.0, 10.0))
        depth_image = ObsTerm(
            func=mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"},
            history_length=8,
            flatten_history_dim=True,
        )
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            history_length=8,
            flatten_history_dim=True,
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            history_length=8,
            flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            history_length=8,
            flatten_history_dim=True,
        )
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={"command_name": "motion"},
            history_length=8,
            flatten_history_dim=True,
        )
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={"command_name": "motion"},
            history_length=8,
            flatten_history_dim=True,
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            history_length=8,
            flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=8,
            flatten_history_dim=True,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            history_length=8,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            history_length=8,
            flatten_history_dim=True,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            history_length=8,
            flatten_history_dim=True,
            clip=(-10.0, 10.0),
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
            history_length=8,
            flatten_history_dim=True,
        )
        depth_image = ObsTerm(
            func=mdp.delayed_visualizable_image,
            params={
                "data_type": "distance_to_image_plane_noised_history",
                "sensor_cfg": SceneEntityCfg("camera"),
                "history_skip_frames": 5,
                "num_output_frames": 8,
                "delayed_frame_ranges": (0, 1),
                "debug_vis": False,
            },
            noise=None,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

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

    randomize_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
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

    randomize_camera_offset = EventTerm(
        func=mdp.randomize_camera_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("camera"),
            "offset_pose_ranges": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.03, 0.03),
                "roll": (-math.radians(3), math.radians(3)),
                "pitch": (-math.radians(3), math.radians(3)),
                "yaw": (-math.radians(3), math.radians(3)),
            },
            "distribution": "uniform",
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box_object"),
            "mass_distribution_params": (0.1, 1.0),
            "operation": "abs",
        },
    )

    randomize_object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box_object"),
            "static_friction_range": (0.4, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP (aligned with beyondmimic, plus undesired_contacts)."""

    # Base
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # Tracking
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    object_global_pos = RewTerm(
        func=mdp.object_global_position_error_exp,
        weight=3.0,
        params={"command_name": "motion", "object_name": "box_object", "std": 0.2},
    )
    object_global_ori = RewTerm(
        func=mdp.object_global_orientation_error_exp,
        weight=2.0,
        params={"command_name": "motion", "object_name": "box_object", "std": 0.4},
    )

    # Others (beyondmimic disables this; keep enabled for HOI)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!.*ankle_roll.*$)"
                    r"(?!.*elbow.*$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class PerceptiveHoiEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for perceptive human-object interaction imitation learning."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
