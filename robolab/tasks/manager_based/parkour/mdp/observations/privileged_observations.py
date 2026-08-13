# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared privileged limb/contact observation helpers for parkour critics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def body_lin_vel_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return selected body linear velocities expressed in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_vel_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids]
    root_quat_w = robot.data.root_quat_w.unsqueeze(1).expand(-1, body_vel_w.shape[1], -1)
    return math_utils.quat_apply_inverse(root_quat_w, body_vel_w).reshape(env.num_envs, -1)


def contact_states(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return binary contact states for the selected sensor bodies."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()
