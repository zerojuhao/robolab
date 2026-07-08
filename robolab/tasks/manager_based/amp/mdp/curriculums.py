# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .events import get_assist_force_buffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def force_level(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    reward_term_name: str,
    initial_force: float = 200.0,
    force_decrement: float = 10.0,
):
    """Reduce assist force when the robot consistently achieves the stand-up height reward."""
    assist_force = get_assist_force_buffer(env, initial_force=initial_force)
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.6 * reward_term_cfg.weight:
        assist_force[env_ids] = (assist_force[env_ids] - force_decrement).clamp(min=0.0)
    return torch.mean(assist_force)
