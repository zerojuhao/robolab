
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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from robolab.tasks.manager_based.perceptive.mdp.commands import MotionCommand
from robolab.tasks.manager_based.perceptive.mdp.rewards import _get_body_indexes
from robolab.utils.obj_terrain import MeshXYFootprint, build_mesh_xy_footprint

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg


class mesh_terrain_out_of_bounds(ManagerTermBase):
    """Terminate when the robot XY position leaves the OBJ terrain footprint.

    Unlike the rectangular ``terrain_out_of_bounds`` term, this rasterizes the mesh
    footprint with downward ray casting and checks whether the robot root XY lies on
    the terrain surface projection.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: RigidObject | Articulation = env.scene[self._asset_cfg.name]
        self._footprint: MeshXYFootprint | None = None
        self._grid: torch.Tensor | None = None

        mesh_path = cfg.params.get("mesh_path")
        if mesh_path and env.scene.cfg.terrain.terrain_type != "plane":
            resolution = float(cfg.params.get("resolution", 0.02))
            distance_buffer = float(cfg.params.get("distance_buffer", 0.1))
            self._footprint = build_mesh_xy_footprint(
                mesh_path,
                resolution=resolution,
                distance_buffer=distance_buffer,
            )
            self._grid = torch.as_tensor(self._footprint.grid, device=env.device, dtype=torch.bool)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        mesh_path: str | None = None,
        distance_buffer: float = 0.1,
        resolution: float = 0.02,
    ) -> torch.Tensor:
        del asset_cfg, mesh_path, distance_buffer, resolution

        if env.scene.cfg.terrain.terrain_type == "plane" or self._grid is None or self._footprint is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        xy = self._asset.data.root_pos_w[:, :2]
        footprint = self._footprint
        cell_x = torch.floor((xy[:, 0] - footprint.origin[0]) / footprint.resolution).long()
        cell_y = torch.floor((xy[:, 1] - footprint.origin[1]) / footprint.resolution).long()

        out_of_bounds = (cell_x < 0) | (cell_y < 0) | (cell_x >= footprint.width) | (cell_y >= footprint.height)
        valid = ~out_of_bounds
        on_terrain = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if torch.any(valid):
            on_terrain[valid] = self._grid[cell_y[valid], cell_x[valid]]
        return out_of_bounds | ~on_terrain


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
