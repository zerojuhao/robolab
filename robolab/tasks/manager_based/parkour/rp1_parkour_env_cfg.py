import copy
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import PR1_LINKS, RP1_3_CFG
from robolab.sensors import get_link_prim_targets
from robolab.tasks.manager_based.parkour.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg
from robolab.sensors import Grid3dPointsGeneratorCfg, NoisyGroupedRayCasterCameraCfg, VolumePointsCfg

# Must match lab_key_body_names in robolab/scripts/tools/retarget/config/rp1.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
]

RP1_3_CFG.init_state.pos = (0.0, 0.0, 0.85)
AMP_NUM_STEPS = 3

# Shared with feet_volume_points and volume_points_penetration reward (same object so shoe / cfg edits stay in sync).
FEET_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.11,
    x_max=0.13,
    x_num=25,
    y_min=-0.04,
    y_max=0.04,
    y_num=9,
    z_min=-0.05,
    z_max=-0.03,
    z_num=3,
)
KNEE_VOLUME_POINTS_GRID = Grid3dPointsGeneratorCfg(
    x_min=-0.02,
    x_max=0.09,
    x_num=12,
    y_min=-0.04,
    y_max=0.04,
    y_num=9,
    z_min=-0.3,
    z_max=0.0,
    z_num=31,
)

ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class RP1ParkourEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = RP1_3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.feet_volume_points.points_generator = FEET_VOLUME_POINTS_GRID
        self.scene.knee_volume_points.points_generator = KNEE_VOLUME_POINTS_GRID
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"
        self.scene.camera.offset.pos = (0.09175, 0.011, 0.3982)
        self.scene.camera.offset.rot = (0.866, 0.0, 0.5, 0.0)
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(PR1_LINKS))
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "rp1_lab"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "36_01": 1,
            "36_11": 1,
            "114_09": 1,
            "A1-_Stand_stageii": 1,
            "B4_-_Stand_to_Walk_backwards_stageii": 1,
            "B9_-__Walk_turn_left_90_stageii": 1,
            "B10_-__Walk_turn_left_45_stageii": 1,
            "B13_-__Walk_turn_right_90_stageii": 1,
            "B14_-__Walk_turn_right_45_t2_stageii": 1,
            "B15_-__Walk_turn_around_stageii": 1,
            "move_back": 1,
            "move_l": 1,
            "move_r": 1,
            "turn_l": 1,
            "turn_r": 1,
        }
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS
        self.observations.disc.history_length = AMP_NUM_STEPS
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=KEY_BODY_NAMES,
                preserve_order=True,
            )
        }

        self.rewards.rewards.rpo_thigh_yaw_joint_sign_penalty = None
        self.rewards.rewards.feet_close_xy_gauss.params["threshold"] = 0.20
        self.rewards.rewards.joint_deviation_upper_body.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint", "waist_.*_joint"])
        self.rewards.rewards.pelvis_orientation_l2.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")
        self.rewards.rewards.pelvis_ang_vel_xy_l2.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")

        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=["waist_yaw_link"])

        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["base_link", "waist_yaw_link"])
        self.events.randomize_rigid_body_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["base_link", "waist_yaw_link"])


@configclass
class RP1ParkourEnvCfg_PLAY(RP1ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None

        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.debug_vis = True
        self.scene.feet_volume_points.debug_vis = True
        self.scene.knee_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
