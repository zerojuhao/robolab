# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
"""Script to play RPO-Perceptive checkpoints (RSL-RL, EncoderActorCritic + depth encoder)."""

import argparse
import os
import sys
import time
import types

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play RPO-Perceptive RL agent (RSL-RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import robolab.tasks  # noqa: F401

from robolab.assets import ISAAC_DATA_DIR
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

RPO_POPSICLE_URDF = os.path.join(ISAAC_DATA_DIR, "robots/roboparty/rpo/urdf/rpo_popsicle.urdf")


def _configure_robot_reference(env_cfg) -> None:
    """Spawn a ghost articulation for dataset motion visualization."""
    env_cfg.scene.replicate_physics = False
    robot_cfg = env_cfg.scene.robot
    env_cfg.scene.robot_reference = robot_cfg.replace(
        prim_path="{ENV_REGEX_NS}/RobotReference",
        actuators={},
        spawn=robot_cfg.spawn.replace(
            asset_path=RPO_POPSICLE_URDF,
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=robot_cfg.spawn.rigid_props.replace(disable_gravity=True),
        ),
    )

    motion_cmd = env_cfg.commands.motion
    motion_cmd.pose_range = {key: (0.0, 0.0) for key in motion_cmd.pose_range}
    motion_cmd.velocity_range = {key: (0.0, 0.0) for key in motion_cmd.velocity_range}
    motion_cmd.joint_position_range = (0.0, 0.0)


def _install_motion_start_from_zero(env) -> None:
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
    print("[INFO] Play: motion starts from dataset frame 0.")


def _sync_robot_reference(env) -> None:
    """Drive ghost articulation from the current motion-command frame."""
    unwrapped = env.unwrapped
    if "robot_reference" not in unwrapped.scene.articulations:
        return

    robot_ref = unwrapped.scene["robot_reference"]
    if not robot_ref.is_initialized:
        return

    motion_cmd = unwrapped.command_manager.get_term("motion")
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)

    root_pos = motion_cmd.body_pos_w[:, 0]
    root_ori = motion_cmd.body_quat_w[:, 0]
    zero_root_vel = torch.zeros(unwrapped.num_envs, 3, device=unwrapped.device)

    robot_ref.write_joint_state_to_sim(
        motion_cmd.joint_pos,
        torch.zeros_like(motion_cmd.joint_vel),
        env_ids=env_ids,
    )
    robot_ref.write_root_state_to_sim(
        torch.cat([root_pos, root_ori, zero_root_vel, zero_root_vel], dim=-1),
        env_ids=env_ids,
    )
    robot_ref.write_data_to_sim()


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
    _configure_robot_reference(env_cfg)

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
    _install_motion_start_from_zero(env)
    _sync_robot_reference(env)

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
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            _sync_robot_reference(env)
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
