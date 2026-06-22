
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from robolab.tasks.manager_based.parkour.mdp.randomization import (
    randomize_camera_offsets as _randomize_camera_offsets,
)
from robolab.tasks.manager_based.perceptive_hoi.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_camera_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
) -> None:
    """Randomize depth camera extrinsics to mimic calibration / installation error."""
    _randomize_camera_offsets(
        env,
        env_ids,
        asset_cfg,
        offset_pose_ranges,
        distribution=distribution,
    )


def reset_rigid_object_state_by_reference(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | Sequence[int],
    command_name: str,
    object_name: str,
    invalid_object_pos: tuple[float, float, float] = (0.0, 0.0, -1.0),
):
    """Reset rigid object pose/velocity from motion reference at the current command time step."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    elif env_ids.device != env.device:
        env_ids = env_ids.to(device=env.device)

    if len(env_ids) == 0:
        return

    command: MotionCommand = env.command_manager.get_term(command_name)
    obj = env.scene[object_name]

    default_pose = torch.zeros(len(env_ids), 7, device=env.device)
    default_pose[:, :3] = torch.tensor(invalid_object_pos, device=env.device)
    default_pose[:, 3] = 1.0
    obj.write_root_pose_to_sim(default_pose, env_ids=env_ids)

    if not command.motion.has_object:
        return

    time_steps = command.time_steps[env_ids]
    object_pos = command.motion.object_pos_w[time_steps]
    object_quat = command.motion.object_quat_w[time_steps]
    object_pos = object_pos + env.scene.env_origins[env_ids]

    valid_mask = object_pos.abs().sum(dim=-1) > 1e-6
    if valid_mask.any():
        valid_env_ids = env_ids[valid_mask]
        object_pos_valid = object_pos[valid_mask]
        object_quat_valid = object_quat[valid_mask]

        pose_range = command.cfg.object_pose_range
        if pose_range:
            range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=env.device)
            rand_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (object_pos_valid.shape[0], 6), device=env.device
            )
            object_pos_valid = object_pos_valid + rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            object_quat_valid = quat_mul(orientations_delta, object_quat_valid)

        obj.write_root_pose_to_sim(
            torch.cat([object_pos_valid, object_quat_valid], dim=-1),
            env_ids=valid_env_ids,
        )
        object_lin_vel = command.motion.object_lin_vel_w[time_steps]
        object_ang_vel = command.motion.object_ang_vel_w[time_steps]
        obj.write_root_velocity_to_sim(
            torch.cat([object_lin_vel[valid_mask], object_ang_vel[valid_mask]], dim=-1),
            env_ids=valid_env_ids,
        )


def update_rigid_object_state_by_reference(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    object_name: str,
):
    """Update rigid object pose from motion reference each control step."""
    del env_ids  # interval events may pass a subset; object pose must stay in sync for all envs
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.motion.has_object:
        return

    object_pos = command.object_pos_w
    object_quat = command.object_quat_w
    if object_pos is None or object_quat is None:
        return

    obj = env.scene[object_name]
    obj.write_root_pose_to_sim(torch.cat([object_pos, object_quat], dim=-1))
