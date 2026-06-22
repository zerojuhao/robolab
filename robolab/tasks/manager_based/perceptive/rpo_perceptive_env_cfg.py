# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import RPO_CFG, RPO_LINKS
from robolab.sensors import get_link_prim_targets
from robolab.tasks.manager_based.perceptive.perceptive_env_cfg import PerceptiveEnvCfg
from robolab.utils.obj_terrain import configure_single_obj_terrain


@configclass
class RPOPerceptiveEnvCfg(PerceptiveEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        terrain_obj_path = os.path.join(
            ROBOLAB_ROOT_DIR,
            "data",
            "motions",
            "rpo_bm",
            "beyond_reverse_vault_003_aug001_dm_aug8_terrain.obj",
        )
        configure_single_obj_terrain(self.scene.terrain, terrain_obj_path, direct=True)
        self.terminations.out_of_border.params["mesh_path"] = terrain_obj_path

        # Compact mesh: keep every env at the same world patch on the terrain.
        self.scene.env_spacing = 0.0

        self.scene.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(RPO_LINKS))
        self.commands.motion.motion_file = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rpo_bm", "beyond_reverse_vault_003_aug001_dm_aug8.npz"
        )
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
