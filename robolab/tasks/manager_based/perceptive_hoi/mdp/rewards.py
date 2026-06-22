
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_error_magnitude

from robolab.tasks.manager_based.perceptive_hoi.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_reference_valid(command: MotionCommand) -> torch.Tensor:
    """True for envs where the motion clip provides a valid object pose."""
    if not command.motion.has_object:
        return torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
    object_pos = command.object_pos_w
    if object_pos is None:
        return torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
    return object_pos.abs().sum(dim=-1) > 1e-6


def object_global_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_name: str,
    std: float,
) -> torch.Tensor:
    """Reward tracking the simulated object position to the reference trajectory."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_pos = command.object_pos_w
    if ref_pos is None:
        return torch.zeros(env.num_envs, device=env.device)

    obj = env.scene[object_name]
    error = torch.sum(torch.square(ref_pos - obj.data.root_pos_w), dim=-1)
    reward = torch.exp(-error / std**2)
    return reward * _object_reference_valid(command)


def object_global_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_name: str,
    std: float,
) -> torch.Tensor:
    """Reward tracking the simulated object orientation to the reference trajectory."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_quat = command.object_quat_w
    if ref_quat is None:
        return torch.zeros(env.num_envs, device=env.device)

    obj = env.scene[object_name]
    error = quat_error_magnitude(ref_quat, obj.data.root_quat_w) ** 2
    reward = torch.exp(-error / std**2)
    return reward * _object_reference_valid(command)
