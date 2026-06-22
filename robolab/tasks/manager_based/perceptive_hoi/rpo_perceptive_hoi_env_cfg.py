
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import RPO_CFG, RPO_LINKS
from robolab.sensors import get_link_prim_targets
from robolab.tasks.manager_based.perceptive_hoi.perceptive_hoi_env_cfg import PerceptiveHoiEnvCfg
from robolab.utils.hoi_object import make_hoi_object_cfg

RPO_BM_MOTION_DIR = os.path.join(ROBOLAB_ROOT_DIR, "data", "motions", "rpo_bm")
LARGEBOX_MOTION_FILE = os.path.join(RPO_BM_MOTION_DIR, "sub10_largebox_000.npz")
LARGEBOX_MESH_FILE = os.path.join(RPO_BM_MOTION_DIR, "largebox_cleaned_simplified.obj")


@configclass
class RPOPerceptiveHoiEnvCfg(PerceptiveHoiEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.box_object = make_hoi_object_cfg(
            LARGEBOX_MESH_FILE,
            prim_path="{ENV_REGEX_NS}/box_object",
            default_mass=1.0,
            kinematic=False,
        )
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(RPO_LINKS))
        self.scene.camera.mesh_prim_paths.append(
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/box_object")
        )

        self.commands.motion.motion_file = LARGEBOX_MOTION_FILE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            # 'left_thigh_yaw_link', 
            # 'right_thigh_yaw_link', 
            "base_link",
            'torso_link', 
            'left_thigh_roll_link', 
            'right_thigh_roll_link', 
            # 'left_arm_pitch_link', 
            # 'right_arm_pitch_link', 
            'left_thigh_pitch_link', 
            'right_thigh_pitch_link', 
            # 'left_arm_roll_link', 
            # 'right_arm_roll_link', 
            'left_knee_link', 
            'right_knee_link', 
            'left_arm_yaw_link', 
            'right_arm_yaw_link', 
            # 'left_ankle_pitch_link', 
            # 'right_ankle_pitch_link', 
            # 'left_elbow_pitch_link', 
            # 'right_elbow_pitch_link', 
            'left_ankle_roll_link', 
            'right_ankle_roll_link', 
            'left_elbow_yaw_link', 
            'right_elbow_yaw_link',
        ]

        self.episode_length_s = 10.0
