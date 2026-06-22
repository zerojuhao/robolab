# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""Script to play RPO-PerceptiveHoi checkpoints (RSL-RL, EncoderActorCritic + depth encoder)."""

import argparse
import os
import sys
import time
import types

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play RPO-PerceptiveHoi RL agent (RSL-RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="RPO-PerceptiveHoi",
    help="Name of the task (default: RPO-PerceptiveHoi).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--exportonnx",
    action="store_true",
    default=False,
    help="Export EncoderActorCritic policy as separate ONNX files (depth encoder + actor).",
)
parser.add_argument(
    "--ghost_lateral_offset",
    type=float,
    default=1.0,
    help="World +Y offset (m) for reference ghosts; trajectory stays parallel to npz replay.",
)
parser.add_argument("--no_ghost", action="store_true", default=False, help="Disable reference ghosts during play.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import robolab.tasks  # noqa: F401

from robolab.assets import ISAAC_DATA_DIR
from robolab.tasks.manager_based.perceptive_hoi.rpo_perceptive_hoi_env_cfg import LARGEBOX_MESH_FILE
from robolab.utils.hoi_object import make_hoi_object_cfg
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

RPO_POPSICLE_URDF = os.path.join(ISAAC_DATA_DIR, "robots/roboparty/rpo/urdf/rpo_popsicle.urdf")
_NO_COLLISION = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
# opacity must stay 1.0 for mesh ghosts; partial opacity often renders invisible while still casting shadows
_GHOST_BOX_VISUAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.3, 0.85, 1.0),
    emissive_color=(0.05, 0.12, 0.2),
    opacity=1.0,
    roughness=0.35,
)


def _fixed_world_offset(num_envs: int, offset_m: float, device: torch.device) -> torch.Tensor:
    """Constant world-frame offset (env +Y), parallel to csv_to_npz_hoi replay."""
    if offset_m == 0.0:
        return torch.zeros(num_envs, 3, device=device)
    offset = torch.zeros(num_envs, 3, device=device)
    offset[:, 1] = offset_m
    return offset


def _reference_poses(motion_cmd, env_origins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw npz reference poses (same source as csv_to_npz export), with env origin."""
    ts = motion_cmd.time_steps
    robot_root_pos = motion_cmd.motion.body_pos_w[ts, 0] + env_origins
    robot_root_quat = motion_cmd.motion.body_quat_w[ts, 0]
    object_pos = motion_cmd.motion.object_pos_w[ts] + env_origins
    object_quat = motion_cmd.motion.object_quat_w[ts]
    return robot_root_pos, robot_root_quat, object_pos, object_quat


def _freeze_motion_ranges(motion_cmd) -> None:
    """Disable motion/object pose randomization for deterministic play."""
    motion_cmd.pose_range = {key: (0.0, 0.0) for key in motion_cmd.pose_range}
    motion_cmd.velocity_range = {key: (0.0, 0.0) for key in motion_cmd.velocity_range}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.object_pose_range = {key: (0.0, 0.0) for key in motion_cmd.object_pose_range}


def _make_box_ghost_cfg() -> RigidObjectCfg:
    cfg = make_hoi_object_cfg(
        LARGEBOX_MESH_FILE,
        prim_path="{ENV_REGEX_NS}/box_object_reference",
        kinematic=True,
        diffuse_color=(0.25, 0.75, 1.0),
    )
    return cfg.replace(
        spawn=cfg.spawn.replace(collision_props=_NO_COLLISION, visual_material=_GHOST_BOX_VISUAL),
    )


def _make_robot_ghost_cfg(robot_cfg) -> ArticulationCfg:
    return robot_cfg.replace(
        prim_path="{ENV_REGEX_NS}/RobotReference",
        actuators={},
        spawn=robot_cfg.spawn.replace(
            asset_path=RPO_POPSICLE_URDF,
            activate_contact_sensors=False,
            collision_props=_NO_COLLISION,
            rigid_props=robot_cfg.spawn.rigid_props.replace(disable_gravity=True),
        ),
    )


def _configure_reference_ghosts(env_cfg, lateral_offset_m: float) -> None:
    """Spawn side-offset robot and box reference ghosts (visual only, no collision)."""
    env_cfg.scene = env_cfg.scene.replace(
        replicate_physics=False,
        robot_reference=_make_robot_ghost_cfg(env_cfg.scene.robot),
        box_object_reference=_make_box_ghost_cfg(),
    )
    _freeze_motion_ranges(env_cfg.commands.motion)
    print(f"[INFO] Play ghosts: reference offset +{lateral_offset_m:.2f} m in world +Y (parallel to npz replay).")


def _configure_play_events(env_cfg) -> None:
    if hasattr(env_cfg, "events"):
        env_cfg.events.randomize_object_mass = None
        env_cfg.events.randomize_camera_offset = None


def _install_motion_start_from_zero(env, ghost_offset: float | None = None) -> None:
    """Always sample motion frame 0 on episode reset."""
    unwrapped = env.unwrapped
    motion_cmd = unwrapped.command_manager.get_term("motion")

    def _adaptive_from_start(self, env_ids):
        if len(env_ids) == 0:
            return
        self.time_steps[env_ids] = 0

    motion_cmd._adaptive_sampling = types.MethodType(_adaptive_from_start, motion_cmd)
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
    motion_cmd._resample_command(env_ids)
    if ghost_offset is not None:
        _sync_reference_ghosts(env, ghost_offset)
    print("[INFO] Play: motion starts from dataset frame 0.")


def _sync_robot_ghost(unwrapped, motion_cmd, offset_w: torch.Tensor) -> None:
    if "robot_reference" not in unwrapped.scene.articulations:
        return
    robot_ref = unwrapped.scene["robot_reference"]
    if not robot_ref.is_initialized:
        return

    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
    zero_vel = torch.zeros(unwrapped.num_envs, 3, device=unwrapped.device)
    root_pos, root_ori, _, _ = _reference_poses(motion_cmd, unwrapped.scene.env_origins)
    root_pos = root_pos + offset_w

    robot_ref.write_joint_state_to_sim(
        motion_cmd.joint_pos,
        torch.zeros_like(motion_cmd.joint_vel),
        env_ids=env_ids,
    )
    robot_ref.write_root_state_to_sim(
        torch.cat([root_pos, root_ori, zero_vel, zero_vel], dim=-1),
        env_ids=env_ids,
    )
    robot_ref.write_data_to_sim()


def _sync_box_ghost(unwrapped, motion_cmd, offset_w: torch.Tensor) -> None:
    if not motion_cmd.motion.has_object:
        return
    try:
        obj_ref = unwrapped.scene["box_object_reference"]
    except KeyError:
        return
    if not obj_ref.is_initialized:
        return

    _, _, object_pos, object_quat = _reference_poses(motion_cmd, unwrapped.scene.env_origins)
    obj_ref.write_root_pose_to_sim(torch.cat([object_pos + offset_w, object_quat], dim=-1))
    obj_ref.write_root_velocity_to_sim(torch.zeros(unwrapped.num_envs, 6, device=unwrapped.device))


def _sync_reference_ghosts(env, lateral_offset_m: float) -> None:
    unwrapped = env.unwrapped
    motion_cmd = unwrapped.command_manager.get_term("motion")
    offset_w = _fixed_world_offset(unwrapped.num_envs, lateral_offset_m, unwrapped.device)
    _sync_robot_ghost(unwrapped, motion_cmd, offset_w)
    _sync_box_ghost(unwrapped, motion_cmd, offset_w)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    if version.parse(installed_version) < version.parse("5.0.0"):
        for key in ("optimizer", "share_cnn_encoders"):
            if hasattr(agent_cfg.algorithm, key):
                delattr(agent_cfg.algorithm, key)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False
    use_ghosts = not args_cli.no_ghost
    if use_ghosts:
        _configure_reference_ghosts(env_cfg, args_cli.ghost_lateral_offset)
    _configure_play_events(env_cfg)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    ghost_offset = args_cli.ghost_lateral_offset if use_ghosts else None
    if use_ghosts and "box_object_reference" not in env.unwrapped.scene.rigid_objects:
        raise RuntimeError(
            "box_object_reference was not spawned. "
            f"Available rigid objects: {list(env.unwrapped.scene.rigid_objects.keys())}"
        )
    _install_motion_start_from_zero(env, ghost_offset)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    torch_policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    export_model_dir = os.path.join(log_dir, "exported")
    if args_cli.exportonnx:
        assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
        if not hasattr(policy_nn, "export_as_onnx"):
            raise AttributeError(
                "export_as_onnx is missing on the policy module; use EncoderActorCritic for perceptive ONNX export."
            )
        os.makedirs(export_model_dir, exist_ok=True)
        obs = env.get_observations()
        policy_nn.export_as_onnx(obs, export_model_dir)
    else:
        obs = env.get_observations()

    policy = torch_policy
    dt = env.unwrapped.step_dt
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if ghost_offset is not None:
                _sync_reference_ghosts(env, ghost_offset)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            if ghost_offset is not None:
                _sync_reference_ghosts(env, ghost_offset)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
