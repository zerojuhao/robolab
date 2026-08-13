"""Contact and terrain support evaluation for imagined footholds."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from isaaclab.utils import configclass
from isaaclab.utils.warp import raycast_mesh

from ..terrain_family import (
    soft_absolute_clearance_unsupported,
    terrain_foot_point_weights,
)

_MISS_PLANE_Z = -1.0e6
_VALID_PLANE_MIN = -1.0e5
_LEGACY_SUPPORT_KEYS = ("reward_sigma", "disable_slope_family", "slope_family_id")


@configclass
class FootholdSupportCfg:
    """Sole geometry and clearance kernel shared with ``feet_at_plane``."""

    foot_body_names: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
    contact_sensor_name: str = "contact_forces"
    height_scanner_names: list[str] = ["left_height_scanner", "right_height_scanner"]
    height_offset: float = 0.035
    height_tolerance: float = 0.03
    support_transition_width: float = 0.005
    ray_start_height: float = 2.0
    ray_max_distance: float = 10.0


def normalize_foothold_support_cfg(
    cfg: FootholdSupportCfg | Mapping,
) -> FootholdSupportCfg:
    if isinstance(cfg, FootholdSupportCfg):
        return cfg
    if isinstance(cfg, Mapping):
        values = dict(cfg)
        for key in _LEGACY_SUPPORT_KEYS:
            values.pop(key, None)
        return FootholdSupportCfg(**values)
    raise TypeError(f"Unsupported foothold support config type: {type(cfg)!r}")


def _finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _pose_ok(candidate_xy_w: torch.Tensor, foot_yaw: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(candidate_xy_w).all(dim=-1) & torch.isfinite(foot_yaw)


class FootholdSupportEvaluator:
    """Evaluate sole support at imagined and real contacts."""

    def __init__(
        self,
        cfg: FootholdSupportCfg,
        sole_offsets: list[torch.Tensor],
        terrain_mesh,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.sole_offsets = sole_offsets
        self.terrain_mesh = terrain_mesh
        self.device = device
        self._origin_index = torch.tensor(
            [
                int(offsets.square().sum(dim=-1).argmin().item())
                for offsets in sole_offsets
            ],
            dtype=torch.long,
            device=device,
        )

    def _sole_offsets_w(
        self, foot_ids: torch.Tensor, foot_yaws: torch.Tensor
    ) -> torch.Tensor:
        num_pairs = foot_ids.shape[0]
        num_sole_points = self.sole_offsets[0].shape[0]
        offsets = torch.empty(num_pairs, num_sole_points, 2, device=self.device)
        for foot_id in range(2):
            foot_mask = foot_ids == foot_id
            if foot_mask.any():
                offsets[foot_mask] = self.sole_offsets[foot_id]

        cos_foot = torch.cos(foot_yaws).unsqueeze(-1)
        sin_foot = torch.sin(foot_yaws).unsqueeze(-1)
        offset_x_w = cos_foot * offsets[..., 0] - sin_foot * offsets[..., 1]
        offset_y_w = sin_foot * offsets[..., 0] + cos_foot * offsets[..., 1]
        return torch.stack((offset_x_w, offset_y_w), dim=-1)

    def _support_plane_from_heights(
        self, heights: torch.Tensor, foot_ids: torch.Tensor
    ) -> torch.Tensor:
        """Terrain height at the sole origin. Origin misses fall back to max finite height."""
        pair_ids = torch.arange(heights.shape[0], device=self.device)
        origin = heights[pair_ids, self._origin_index[foot_ids]]
        fallback = (
            torch.where(
                torch.isfinite(heights),
                heights,
                torch.full_like(heights, _MISS_PLANE_Z),
            )
            .max(dim=-1)
            .values
        )
        origin_ok = torch.isfinite(origin) & (origin > _VALID_PLANE_MIN)
        return torch.where(origin_ok, origin, fallback)

    def _sole_terrain_point_weights(
        self,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor,
        stairs_weight_min: float,
        stairs_weight_max: float,
    ) -> torch.Tensor:
        num_pairs = foot_ids.shape[0]
        num_points = self.sole_offsets[0].shape[0]
        weights = torch.ones(num_pairs, num_points, device=self.device)
        for foot_id in range(2):
            foot_mask = foot_ids == foot_id
            if not foot_mask.any():
                continue
            weights[foot_mask] = terrain_foot_point_weights(
                self.sole_offsets[foot_id][:, 0],
                family_ids[foot_mask],
                stairs_weight_min,
                stairs_weight_max,
            )
        return weights

    def _geometry_from_xy_b(
        self,
        xy_b: torch.Tensor,
        base_frames: torch.Tensor,
        foot_yaws: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raycast sole heights at predicted base-frame XY.

        ``foot_yaws`` is world yaw with shape ``[num_envs, 2]``.
        """
        frames = base_frames[env_ids]
        foot_yaw = foot_yaws[env_ids, foot_ids]
        cos_base = frames[:, 3]
        sin_base = frames[:, 4]
        xy_w = torch.stack(
            (
                frames[:, 0] + cos_base * xy_b[:, 0] - sin_base * xy_b[:, 1],
                frames[:, 1] + sin_base * xy_b[:, 0] + cos_base * xy_b[:, 1],
            ),
            dim=-1,
        )

        offsets_w = self._sole_offsets_w(foot_ids, foot_yaw)
        num_pairs = xy_w.shape[0]
        num_sole_points = offsets_w.shape[1]
        ray_starts = torch.zeros(num_pairs, num_sole_points, 3, device=self.device)
        ray_starts[..., :2] = xy_w[:, None, :] + offsets_w
        ray_starts[..., 2] = frames[:, 2, None] + self.cfg.ray_start_height
        ray_directions = torch.zeros_like(ray_starts)
        ray_directions[..., 2] = -1.0
        ray_hits, _, _, _ = raycast_mesh(
            ray_starts,
            ray_directions,
            mesh=self.terrain_mesh,
            max_dist=self.cfg.ray_max_distance,
        )
        return xy_w, foot_yaw, ray_hits[..., 2]

    def _contact_terrain_heights(
        self,
        foot_pos_w: torch.Tensor,
        foot_yaws: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> torch.Tensor:
        offsets_w = self._sole_offsets_w(foot_ids, foot_yaws)
        ray_starts = torch.zeros(
            foot_pos_w.shape[0], offsets_w.shape[1], 3, device=self.device
        )
        ray_starts[..., :2] = foot_pos_w[:, None, :2] + offsets_w
        ray_starts[..., 2] = foot_pos_w[:, None, 2] + self.cfg.ray_start_height
        ray_directions = torch.zeros_like(ray_starts)
        ray_directions[..., 2] = -1.0
        ray_hits, _, _, _ = raycast_mesh(
            ray_starts,
            ray_directions,
            mesh=self.terrain_mesh,
            max_dist=self.cfg.ray_max_distance,
        )
        return ray_hits[..., 2]

    def contact_support_ratio(
        self,
        foot_pos_w: torch.Tensor,
        foot_yaws: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
    ) -> torch.Tensor:
        """Supported sole fraction at a real contact pose."""
        point_w = None
        if enable_terrain_foot_weights and family_ids is not None:
            point_w = self._sole_terrain_point_weights(
                foot_ids, family_ids, stairs_weight_min, stairs_weight_max
            )
        unsupported = _finite_or_zero(
            soft_absolute_clearance_unsupported(
                foot_pos_w[:, 2].unsqueeze(-1),
                self._contact_terrain_heights(foot_pos_w, foot_yaws, foot_ids),
                height_offset=self.cfg.height_offset,
                height_tolerance=self.cfg.height_tolerance,
                transition_width=self.cfg.support_transition_width,
                point_weights=point_w,
            )
        )
        return 1.0 - unsupported

    def _support_deficiency_for_xy(
        self,
        xy_b: torch.Tensor,
        base_frames: torch.Tensor,
        foot_yaws: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
    ) -> torch.Tensor:
        xy_w, foot_yaw, heights = self._geometry_from_xy_b(
            xy_b, base_frames, foot_yaws, env_ids, foot_ids
        )
        point_w = None
        if enable_terrain_foot_weights and family_ids is not None:
            point_w = self._sole_terrain_point_weights(
                foot_ids,
                family_ids[env_ids],
                stairs_weight_min,
                stairs_weight_max,
            )
        support_plane = self._support_plane_from_heights(heights, foot_ids)
        foot_z = (support_plane + self.cfg.height_offset).unsqueeze(-1)
        deficiency = soft_absolute_clearance_unsupported(
            foot_z,
            heights,
            height_offset=self.cfg.height_offset,
            height_tolerance=self.cfg.height_tolerance,
            transition_width=self.cfg.support_transition_width,
            point_weights=point_w,
        )
        deficiency = torch.nan_to_num(
            deficiency, nan=1.0, posinf=1.0, neginf=1.0
        ).clamp(0.0, 1.0)
        deficiency = torch.where(
            _pose_ok(xy_w, foot_yaw), deficiency, torch.ones_like(deficiency)
        )
        return _finite_or_zero(deficiency)

    def expected_support_deficiency(
        self,
        candidate_xy_b: torch.Tensor,
        candidate_weights: torch.Tensor,
        base_frames: torch.Tensor,
        foot_yaws: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
        eval_chunk_size: int = 64,
    ) -> torch.Tensor:
        """Expected unsupported fraction over XY quadrature samples.

        ``candidate_xy_b`` is ``[P, G, 2]`` and ``candidate_weights`` is ``[P, G]``.
        ``foot_yaws`` is predicted world yaw ``[num_envs, 2]``.
        """
        num_pairs, num_candidates = candidate_weights.shape
        if num_pairs == 0:
            return torch.zeros(0, device=self.device)
        if candidate_xy_b.shape != (num_pairs, num_candidates, 2):
            raise ValueError(
                "candidate_xy_b must have shape [P, G, 2] matching candidate_weights."
            )
        weights = torch.nan_to_num(
            candidate_weights, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        expected = torch.zeros(num_pairs, device=self.device)
        chunk_size = max(int(eval_chunk_size), 1)
        for start in range(0, num_pairs, chunk_size):
            end = min(start + chunk_size, num_pairs)
            pair_count = end - start
            chunk_env_ids = env_ids[start:end]
            chunk_foot_ids = foot_ids[start:end]
            flat_xy = candidate_xy_b[start:end].reshape(-1, 2)
            flat_env_ids = chunk_env_ids[:, None].expand(-1, num_candidates).reshape(-1)
            flat_foot_ids = chunk_foot_ids[:, None].expand(-1, num_candidates).reshape(-1)
            flat_deficiency = self._support_deficiency_for_xy(
                flat_xy,
                base_frames,
                foot_yaws,
                flat_env_ids,
                flat_foot_ids,
                family_ids=family_ids,
                enable_terrain_foot_weights=enable_terrain_foot_weights,
                stairs_weight_min=stairs_weight_min,
                stairs_weight_max=stairs_weight_max,
            ).view(pair_count, num_candidates)
            expected[start:end] = (weights[start:end] * flat_deficiency).sum(dim=-1)
        return _finite_or_zero(expected).clamp(0.0, 1.0)
