# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import isaaclab.utils.math as math_utils
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from robolab.tasks.manager_based.amp.animation_env import AnimationEnv

_ASSIST_FORCE_ATTR = "get_up_assist_force"


def get_assist_force_buffer(env: ManagerBasedRLEnv, initial_force: float = 200.0) -> torch.Tensor:
    """Return per-env assist force buffer, creating it on first access."""
    if not hasattr(env, _ASSIST_FORCE_ATTR):
        setattr(
            env,
            _ASSIST_FORCE_ATTR,
            torch.full((env.num_envs,), initial_force, device=env.device, dtype=torch.float32),
        )
    return getattr(env, _ASSIST_FORCE_ATTR)


def init_assist_force(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    initial_force: float = 200.0,
):
    """Initialize assist force buffer values (typically on startup or reset)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    buffer = get_assist_force_buffer(env, initial_force=initial_force)
    buffer[env_ids] = initial_force


def _sample_fallen_root_orientation(
    num_envs: int,
    device: torch.device,
    yaw_range: tuple[float, float],
) -> torch.Tensor:
    """Sample supine/prone orientations with random heading."""
    is_prone = torch.rand(num_envs, device=device) < 0.5
    pitch = torch.where(
        is_prone,
        torch.full((num_envs,), 0.5 * math.pi, device=device),
        torch.full((num_envs,), -0.5 * math.pi, device=device),
    )
    roll = torch.zeros(num_envs, device=device)
    yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (num_envs,), device=device)
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)


def _apply_joint_pos_noise(
    joint_pos: torch.Tensor,
    env_ids: torch.Tensor,
    robot: Articulation,
    noise: float,
    device: torch.device,
) -> torch.Tensor:
    """Add uniform joint noise and clamp to soft joint position limits."""
    joint_pos = joint_pos.clone()
    if noise > 0.0:
        joint_pos += math_utils.sample_uniform(-noise, noise, joint_pos.shape, device=device)
    joint_pos_limits = robot.data.soft_joint_pos_limits[env_ids]
    return joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])


def _sample_motion_times_early_phase(
    motion_loader,
    motion_ids: torch.Tensor,
    max_motion_phase: float,
    device: torch.device,
) -> torch.Tensor:
    durations = motion_loader.get_motion_durations(motion_ids)
    phase = torch.rand(len(motion_ids), device=device) * max_motion_phase
    return phase * durations


def reset_get_up_robust(
    env: AnimationEnv,
    env_ids: torch.Tensor,
    motion_data_term: str = "motion_dataset",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_fallen_prob: float = 0.6,
    fallen_deviated_prob: float = 0.4,
    max_motion_phase: float = 0.3,
    height_offset: float = 0.05,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    yaw_noise: float = 0.35,
    motion_joint_pos_noise: float = 0.1,
    fallen_joint_pos_noise: float = 0.45,
    fallen_root_height_range: tuple[float, float] = (0.15, 0.35),
):
    """Mixed reset: motion RSI on early fallen phase, or prone/supine with deviated default joints."""
    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_reset = len(env_ids)
    device = env.device

    mode_probs = torch.tensor([motion_fallen_prob, fallen_deviated_prob], device=device, dtype=torch.float32)
    mode_probs = mode_probs / mode_probs.sum()
    modes = torch.multinomial(mode_probs, num_reset, replacement=True)

    root_pose = torch.zeros(num_reset, 7, device=device)
    joint_pos = torch.zeros(num_reset, robot.num_joints, device=device)

    motion_mask = modes == 0
    if motion_mask.any():
        local_ids = motion_mask.nonzero(as_tuple=False).squeeze(-1)
        batch_env_ids = env_ids[local_ids]
        batch_size = len(local_ids)

        motion_loader = env.motion_data_manager.get_term(motion_data_term)
        motion_ids = motion_loader.sample_motions(batch_size)
        motion_times = _sample_motion_times_early_phase(motion_loader, motion_ids, max_motion_phase, device)
        motion_state = motion_loader.get_motion_state(motion_ids, motion_times)

        root_pos = motion_state["root_pos_w"].clone()
        root_pos[:, :2] = 0.0
        root_pos[:, 2] += height_offset
        root_pose[motion_mask, :3] = root_pos + env.scene.env_origins[batch_env_ids]

        orientations = motion_state["root_quat"].clone()
        if yaw_noise > 0.0:
            yaw_delta = math_utils.sample_uniform(-yaw_noise, yaw_noise, (batch_size,), device=device)
            delta_quat = math_utils.quat_from_euler_xyz(
                torch.zeros(batch_size, device=device),
                torch.zeros(batch_size, device=device),
                yaw_delta,
            )
            orientations = math_utils.quat_mul(delta_quat, orientations)
        root_pose[motion_mask, 3:7] = orientations

        joint_pos[motion_mask] = _apply_joint_pos_noise(
            motion_state["dof_pos"], batch_env_ids, robot, motion_joint_pos_noise, device
        )

    fallen_mask = modes == 1
    if fallen_mask.any():
        local_ids = fallen_mask.nonzero(as_tuple=False).squeeze(-1)
        batch_env_ids = env_ids[local_ids]
        batch_size = len(local_ids)

        root_height = math_utils.sample_uniform(
            fallen_root_height_range[0], fallen_root_height_range[1], (batch_size, 1), device=device
        )
        root_pos = torch.cat([torch.zeros(batch_size, 2, device=device), root_height], dim=-1)
        root_pose[fallen_mask, :3] = root_pos + env.scene.env_origins[batch_env_ids]
        root_pose[fallen_mask, 3:7] = _sample_fallen_root_orientation(batch_size, device, yaw_range)

        default_joint_pos = robot.data.default_joint_pos[batch_env_ids]
        joint_pos[fallen_mask] = _apply_joint_pos_noise(
            default_joint_pos, batch_env_ids, robot, fallen_joint_pos_noise, device
        )

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(num_reset, 6, device=device), env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)


def reset_get_up_from_motion(
    env: AnimationEnv,
    env_ids: torch.Tensor,
    **kwargs,
):
    """Backward-compatible wrapper around :func:`reset_get_up_robust` with motion-only settings."""
    kwargs.setdefault("motion_fallen_prob", 1.0)
    kwargs.setdefault("fallen_deviated_prob", 0.0)
    reset_get_up_robust(env, env_ids, **kwargs)


def apply_force(
    env: AnimationEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    initial_force: float = 200.0,
):
    """Apply upward assist force from env buffer on the configured body."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    size = (len(env_ids), num_bodies, 3)
    forces = torch.zeros(size, device=asset.device)
    torques = torch.zeros(size, device=asset.device)
    assist_force = get_assist_force_buffer(env, initial_force=initial_force)
    forces[:, 0, 2] = assist_force[env_ids]
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids, is_global=True)
