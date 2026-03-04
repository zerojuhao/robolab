# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robolab.tasks.manager_based.amp.mdp as mdp
from robolab.tasks.manager_based.amp.managers import MotionDataTermCfg
from robolab.tasks.manager_based.amp.amp_env_cfg import AmpEnvCfg, MotionDataCfg

import isaaclab.terrains as terrain_gen

##
# Pre-defined configs
##

from robolab.assets.robots.roboparty import ATOM01_CFG
from robolab import ROBOLAB_ROOT_DIR

KEY_BODY_NAMES = [
    "left_ankle_roll_link", 
    "right_ankle_roll_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
    "left_arm_roll_link",
    "right_arm_roll_link"
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 3

@configclass
class Atom01AmpRewards():
    """Reward terms for the MDP."""

    # -- Task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0, params={"command_name": "base_velocity", "std": 0.5}
    )
    
    # -- Alive
    alive = RewTerm(func=mdp.is_alive, weight=0)
    
    # -- Base Link
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0)

    # -- Joint
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0)
    smoothness_1 = RewTerm(func=mdp.smoothness_1, weight=0)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0)
    joint_energy = RewTerm(func=mdp.joint_energy, weight=0)
    joint_regularization = RewTerm(func=mdp.joint_deviation_l1, weight=0)
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
    )
        
    # -- Feet
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    
    sound_suppression = RewTerm(
        func=mdp.sound_suppression_acc_per_foot,
        weight=0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*_ankle_roll_link",
            ),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class Atom01AmpEnvCfg(AmpEnvCfg):
    rewards: Atom01AmpRewards = Atom01AmpRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # ------------------------------------------------------
        # Scene
        # ------------------------------------------------------
        self.scene.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "atom01_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights={
            
            # CMU
            "127_04": 1, # walk to run
            "127_06": 1, # run 
            
            #ACCAD
            "A1-_Stand_stageii": 1,
            
            "B9_-__Walk_turn_left_90_stageii":1,
            "B10_-__Walk_turn_left_45_stageii":1,
            "B13_-__Walk_turn_right_90_stageii":1,
            "B14_-__Walk_turn_right_45_t2_stageii":1,
            "B15_-__Walk_turn_around_stageii_walk":1,
            
            "C12_-_run_turn_left_45_stageii":1,
            "C17_-_run_change_direction_stageii":1,
            
            # GVHMR
            "move_back":1,
            "move_l":1,
            "move_r":1,
            "turn_l":1,
            "turn_r":1,
        }
        
        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------
                
        # discriminator observations
        
        # self.observations.disc.key_body_pos_b.params = {
        #     "asset_cfg": SceneEntityCfg(
        #         name="robot", 
        #         body_names=KEY_BODY_NAMES, 
        #         preserve_order=True
        #     )
        # }
        self.observations.disc.history_length = AMP_NUM_STEPS
        
        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------
        # task
        self.rewards.track_lin_vel_xy_exp.weight = 1.25
        self.rewards.track_ang_vel_z_exp.weight = 1.25
        self.rewards.alive.weight = 0.15
        
        # base
        # self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.0
        
        # joint
        self.rewards.joint_vel_l2.weight = -2e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_energy.weight = -1e-4
        self.rewards.joint_torques_l2.weight = -1e-5
        
        # feet
        self.rewards.feet_slide.weight = -0.1
        self.rewards.sound_suppression.weight = -5e-5


        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["(?!.*ankle.*).*"],  # exclude ankle links
        )
        
        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
                
        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        
        
        
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            ".*_thigh_.*_link", "base_link", ".*_arm_.*_link", ".*_elbow_.*_link",
        ]
        if self.__class__.__name__ == "Atom01AmpEnvCfg":
            self.disable_zero_weight_rewards()
            
            
@configclass
class Atom01AmpEnvCfg_PLAY(Atom01AmpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.push_robot = None