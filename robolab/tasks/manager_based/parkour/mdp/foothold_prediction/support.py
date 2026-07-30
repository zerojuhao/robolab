"""Environment-owned contact and terrain support configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from isaaclab.utils import configclass
from isaaclab.utils.warp import raycast_mesh

from ..terrain_family import (
    SLOPE_FAMILY_ID,
    soft_absolute_clearance_unsupported,
    terrain_foot_point_weights,
)

# Sentinel used when every sole ray misses; must stay below ``_VALID_PLANE_MIN``.
_MISS_PLANE_Z = -1.0e6
_VALID_PLANE_MIN = -1.0e5


@configclass
class FootholdSupportCfg:
    foot_body_names: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
    contact_sensor_name: str = "contact_forces"
    height_scanner_names: list[str] = ["left_height_scanner", "right_height_scanner"]
    # Absolute clearance kernel shared with ``feet_at_plane`` / imagined guidance.
    height_offset: float = 0.03
    height_tolerance: float = 0.03
    support_transition_width: float = 0.005
    reward_sigma: float = 0.0625
    ray_start_height: float = 2.0
    ray_max_distance: float = 10.0
    disable_slope_family: bool = True
    slope_family_id: int = SLOPE_FAMILY_ID


def normalize_foothold_support_cfg(
    cfg: FootholdSupportCfg | Mapping,
) -> FootholdSupportCfg:
    if isinstance(cfg, FootholdSupportCfg):
        return cfg
    if isinstance(cfg, Mapping):
        return FootholdSupportCfg(**dict(cfg))
    raise TypeError(f"Unsupported foothold support config type: {type(cfg)!r}")


def _finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with 0 (safe default for rewards and weights)."""
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _is_valid_plane(support_plane: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(support_plane) & (support_plane > _VALID_PLANE_MIN)


def _masked_zero(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Zero invalid entries without ``values * valid`` (NaN * 0 is still NaN)."""
    while valid.ndim < values.ndim:
        valid = valid.unsqueeze(-1)
    return torch.where(valid, values, torch.zeros_like(values))


def _pose_ok(candidate_xy_w: torch.Tensor, foot_yaw: torch.Tensor) -> torch.Tensor:
    """Per-candidate mask: world XY and foot yaw are all finite."""
    return torch.isfinite(candidate_xy_w).all(dim=-1) & torch.isfinite(
        foot_yaw
    ).unsqueeze(-1)


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

    def _sole_offsets_w(
        self, foot_ids: torch.Tensor, foot_yaws: torch.Tensor
    ) -> torch.Tensor:
        """Select each foot's sole pattern and rotate it into the world XY frame."""
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

    def _support_plane_from_heights(self, heights: torch.Tensor) -> torch.Tensor:
        """Max finite sole height; misses become ``_MISS_PLANE_Z``."""
        valid = torch.isfinite(heights)
        return (
            torch.where(valid, heights, torch.full_like(heights, _MISS_PLANE_Z))
            .max(dim=-1)
            .values
        )

    def _sole_terrain_point_weights(
        self,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor,
        stairs_weight_min: float,
        stairs_weight_max: float,
    ) -> torch.Tensor:
        """Terrain-dependent sole weights matching ``volume_points_penetration_feet``."""
        num_pairs = foot_ids.shape[0]
        num_points = self.sole_offsets[0].shape[0]
        weights = torch.ones(num_pairs, num_points, device=self.device)
        for foot_id in range(2):
            foot_mask = foot_ids == foot_id
            if not foot_mask.any():
                continue
            local_x = self.sole_offsets[foot_id][:, 0]
            weights[foot_mask] = terrain_foot_point_weights(
                local_x,
                family_ids[foot_mask],
                stairs_weight_min,
                stairs_weight_max,
            )
        return weights


    def _geometry_from_indices(
        self,
        candidate_indices: torch.Tensor,
        residuals_all: torch.Tensor,
        grid_points: torch.Tensor,
        base_frames: torch.Tensor,
        foot_yaws: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raycast sole heights at residual-refined cells. Returns xy, yaw, heights, frames."""
        residuals = residuals_all[env_ids, foot_ids]
        candidate_residuals = torch.gather(
            residuals, 1, candidate_indices.unsqueeze(-1).expand(-1, -1, 2)
        )
        candidate_xy_b = grid_points[candidate_indices] + candidate_residuals
        frames = base_frames[env_ids]
        foot_yaw = foot_yaws[env_ids, foot_ids]

        cos_base = frames[:, 3, None]
        sin_base = frames[:, 4, None]
        candidate_xy_w = torch.stack(
            (
                frames[:, 0, None]
                + cos_base * candidate_xy_b[..., 0]
                - sin_base * candidate_xy_b[..., 1],
                frames[:, 1, None]
                + sin_base * candidate_xy_b[..., 0]
                + cos_base * candidate_xy_b[..., 1],
            ),
            dim=-1,
        )

        offsets_w = self._sole_offsets_w(foot_ids, foot_yaw)
        num_pairs, num_candidates = candidate_xy_w.shape[:2]
        num_sole_points = offsets_w.shape[1]
        ray_starts = torch.zeros(
            num_pairs, num_candidates, num_sole_points, 3, device=self.device
        )
        ray_starts[..., :2] = candidate_xy_w[:, :, None, :] + offsets_w[:, None, :, :]
        ray_starts[..., 2] = frames[:, 2, None, None] + self.cfg.ray_start_height
        ray_directions = torch.zeros_like(ray_starts)
        ray_directions[..., 2] = -1.0

        ray_hits, _, _, _ = raycast_mesh(
            ray_starts.reshape(num_pairs, -1, 3),
            ray_directions.reshape(num_pairs, -1, 3),
            mesh=self.terrain_mesh,
            max_dist=self.cfg.ray_max_distance,
        )
        heights = ray_hits[..., 2].view(num_pairs, num_candidates, num_sole_points)
        return candidate_xy_w, foot_yaw, heights, frames

    def _reachable_candidate_indices(
        self,
        reachable_mask: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return every reachable cell id per foot plus a padding mask."""
        reachable_counts = reachable_mask.sum(dim=-1)
        max_candidates = int(reachable_counts.max().item())
        num_pairs = foot_ids.shape[0]
        candidate_indices = torch.empty(
            num_pairs, max_candidates, dtype=torch.long, device=self.device
        )
        candidate_mask = torch.zeros(
            num_pairs, max_candidates, dtype=torch.bool, device=self.device
        )
        for current_foot_id in range(2):
            pair_mask = foot_ids == current_foot_id
            if not pair_mask.any():
                continue
            cell_ids = reachable_mask[current_foot_id].nonzero(
                as_tuple=False
            ).squeeze(-1)
            count = int(cell_ids.numel())
            candidate_indices[pair_mask, :count] = cell_ids
            candidate_indices[pair_mask, count:] = cell_ids[0]
            candidate_mask[pair_mask, :count] = True
        return candidate_indices, candidate_mask

    def _quality_topk_candidates(
        self,
        probabilities_all: torch.Tensor,
        reachable_mask: torch.Tensor,
        grid_points: torch.Tensor,
        residuals_all: torch.Tensor,
        quality_top_k: int,
        quality_eval_chunk_size: int,
        base_frames: torch.Tensor,
        foot_yaws: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
        point_weights: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Select global support-quality Top-K after evaluating all reachable cells."""
        candidate_indices, candidate_mask = self._reachable_candidate_indices(
            reachable_mask, foot_ids
        )
        probabilities = probabilities_all[env_ids, foot_ids]
        num_pairs, num_candidates = candidate_indices.shape
        quality_top_k = int(quality_top_k)
        chunk_size = int(quality_eval_chunk_size)
        num_sole_points = self.sole_offsets[0].shape[0]

        best_scores = torch.full(
            (num_pairs, quality_top_k), torch.inf, device=self.device
        )
        best_xy_w = torch.zeros(
            num_pairs, quality_top_k, 2, device=self.device
        )
        best_heights = torch.zeros(
            num_pairs, quality_top_k, num_sole_points, device=self.device
        )
        best_probabilities = torch.zeros(
            num_pairs, quality_top_k, device=self.device
        )
        best_deficiency = torch.ones(
            num_pairs, quality_top_k, device=self.device
        )
        foot_yaw = foot_yaws[env_ids, foot_ids]

        for start in range(0, num_candidates, chunk_size):
            end = min(start + chunk_size, num_candidates)
            chunk_indices = candidate_indices[:, start:end]
            chunk_mask = candidate_mask[:, start:end]
            chunk_xy_w, _, chunk_heights, _ = self._geometry_from_indices(
                chunk_indices,
                residuals_all,
                grid_points,
                base_frames,
                foot_yaws,
                env_ids,
                foot_ids,
            )
            chunk_probabilities = torch.gather(
                probabilities, 1, chunk_indices
            )
            support_plane = self._support_plane_from_heights(chunk_heights)
            foot_z = (support_plane + self.cfg.height_offset).unsqueeze(-1)
            chunk_deficiency = soft_absolute_clearance_unsupported(
                foot_z,
                chunk_heights,
                height_offset=self.cfg.height_offset,
                height_tolerance=self.cfg.height_tolerance,
                transition_width=self.cfg.support_transition_width,
                point_weights=point_weights,
            )
            chunk_deficiency = torch.nan_to_num(
                chunk_deficiency, nan=1.0, posinf=1.0, neginf=1.0
            ).clamp(0.0, 1.0)
            chunk_deficiency = torch.where(
                _pose_ok(chunk_xy_w, foot_yaw),
                chunk_deficiency,
                torch.ones_like(chunk_deficiency),
            )
            # Probability only breaks numerical quality ties; it cannot override a
            # meaningfully lower support deficiency.
            quality_tie_break = 1.0e-6 * chunk_probabilities
            chunk_scores = torch.where(
                chunk_mask,
                chunk_deficiency - quality_tie_break,
                torch.full_like(chunk_deficiency, torch.inf),
            )
            merged_scores = torch.cat((best_scores, chunk_scores), dim=1)
            merged_xy_w = torch.cat((best_xy_w, chunk_xy_w), dim=1)
            merged_heights = torch.cat((best_heights, chunk_heights), dim=1)
            merged_probabilities = torch.cat(
                (best_probabilities, chunk_probabilities), dim=1
            )
            merged_deficiency = torch.cat(
                (best_deficiency, chunk_deficiency), dim=1
            )
            selected_indices = torch.topk(
                merged_scores,
                k=quality_top_k,
                dim=-1,
                largest=False,
                sorted=False,
            ).indices
            best_scores = torch.gather(merged_scores, 1, selected_indices)
            best_xy_w = torch.gather(
                merged_xy_w,
                1,
                selected_indices.unsqueeze(-1).expand(-1, -1, 2),
            )
            best_heights = torch.gather(
                merged_heights,
                1,
                selected_indices.unsqueeze(-1).expand(
                    -1, -1, num_sole_points
                ),
            )
            best_probabilities = torch.gather(
                merged_probabilities, 1, selected_indices
            )
            best_deficiency = torch.gather(
                merged_deficiency, 1, selected_indices
            )

        return (
            best_xy_w,
            foot_yaw,
            best_heights,
            best_probabilities,
            best_deficiency,
        )
    def _contact_terrain_heights(
        self,
        foot_pos_w: torch.Tensor,
        foot_yaws: torch.Tensor,
        foot_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Raycast sole-pattern terrain heights at real foot poses."""
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
    ) -> torch.Tensor:
        """Mean per-foot support using the ``feet_at_plane`` clearance kernel (uniform sole weights)."""
        terrain_heights = self._contact_terrain_heights(foot_pos_w, foot_yaws, foot_ids)
        unsupported = _finite_or_zero(
            soft_absolute_clearance_unsupported(
                foot_pos_w[:, 2].unsqueeze(-1),
                terrain_heights,
                height_offset=self.cfg.height_offset,
                height_tolerance=self.cfg.height_tolerance,
                transition_width=self.cfg.support_transition_width,
                point_weights=None,
            )
        )
        return 1.0 - unsupported

    def contact_unsupported_penalty(
        self,
        foot_pos_w: torch.Tensor,
        foot_yaws: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
    ) -> torch.Tensor:
        """Absolute soft clearance unsupported fraction at real foot poses (``[0, 1]``)."""
        terrain_heights = self._contact_terrain_heights(foot_pos_w, foot_yaws, foot_ids)
        point_w = None
        if enable_terrain_foot_weights:
            point_w = self._sole_terrain_point_weights(
                foot_ids,
                family_ids,
                stairs_weight_min,
                stairs_weight_max,
            )
        return _finite_or_zero(
            soft_absolute_clearance_unsupported(
                foot_pos_w[:, 2].unsqueeze(-1),
                terrain_heights,
                height_offset=self.cfg.height_offset,
                height_tolerance=self.cfg.height_tolerance,
                transition_width=self.cfg.support_transition_width,
                point_weights=point_w,
            )
        )

    def support_deficiency(
        self,
        probabilities_all: torch.Tensor,
        reachable_mask: torch.Tensor,
        grid_points: torch.Tensor,
        residuals_all: torch.Tensor,
        quality_top_k: int,
        quality_eval_chunk_size: int,
        base_frames: torch.Tensor,
        foot_yaws: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor | None = None,
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
        unselected_mass_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Score quality Top-K support and penalize probability mass outside it.

        Every reachable cell is evaluated for support quality. The best K cells are
        retained independently of predictor probability, then their raw probability
        mass weights the support deficiency. Probability outside the quality Top-K
        is charged ``unselected_mass_penalty`` (default 1 = fully unsupported).
        """
        point_w = None
        if enable_terrain_foot_weights and family_ids is not None:
            point_w = self._sole_terrain_point_weights(
                foot_ids,
                family_ids[env_ids],
                stairs_weight_min,
                stairs_weight_max,
            )
        (
            _,
            _,
            _,
            selected_p,
            selected_deficiency,
        ) = self._quality_topk_candidates(
            probabilities_all,
            reachable_mask,
            grid_points,
            residuals_all,
            quality_top_k,
            quality_eval_chunk_size,
            base_frames,
            foot_yaws,
            env_ids,
            foot_ids,
            point_weights=point_w,
        )
        evaluated = (selected_p * selected_deficiency).sum(dim=-1)
        selected_mass = selected_p.sum(dim=-1)
        unselected_mass = (1.0 - selected_mass).clamp(0.0, 1.0)
        return _finite_or_zero(
            evaluated + unselected_mass * float(unselected_mass_penalty)
        )


    @staticmethod
    def _terrain_foot_point_weights(
        local_points: torch.Tensor,
        x_min: float,
        x_max: float,
        family_ids: torch.Tensor,
        stairs_weight_min: float,
        stairs_weight_max: float,
    ) -> torch.Tensor:
        """Per-pair point weights matching ``volume_points_penetration_feet``."""
        del x_min, x_max  # derived inside shared helper from local_x
        return terrain_foot_point_weights(
            local_points[:, 0],
            family_ids,
            stairs_weight_min,
            stairs_weight_max,
        )

    def edge_penetration(
        self,
        probabilities_all: torch.Tensor,
        reachable_mask: torch.Tensor,
        grid_points: torch.Tensor,
        residuals_all: torch.Tensor,
        quality_top_k: int,
        quality_eval_chunk_size: int,
        base_frames: torch.Tensor,
        foot_yaws: torch.Tensor,
        env_ids: torch.Tensor,
        foot_ids: torch.Tensor,
        family_ids: torch.Tensor,
        local_volume_points: torch.Tensor,
        volume_x_min: float,
        volume_x_max: float,
        volume_z_min: float,
        virtual_obstacles: Mapping[str, Any],
        enable_terrain_foot_weights: bool = True,
        stairs_weight_min: float = 0.0,
        stairs_weight_max: float = 1.0,
        tolerance: float = 0.0,
    ) -> torch.Tensor:
        """Expected virtual-edge penetration over the support-quality Top-K.

        Places the foot volume-point cloud so the sole bottom rests on the raycast
        support plane, then queries the same virtual obstacles used by
        ``volume_points_penetration_feet``. No velocity term (static imagined pose).
        Candidate selection is identical to ``support_deficiency``.
        """
        support_point_w = None
        if enable_terrain_foot_weights:
            support_point_w = self._sole_terrain_point_weights(
                foot_ids,
                family_ids[env_ids],
                stairs_weight_min,
                stairs_weight_max,
            )
        (
            candidate_xy_w,
            foot_yaw,
            candidate_heights,
            candidate_weights,
            _,
        ) = self._quality_topk_candidates(
            probabilities_all,
            reachable_mask,
            grid_points,
            residuals_all,
            quality_top_k,
            quality_eval_chunk_size,
            base_frames,
            foot_yaws,
            env_ids,
            foot_ids,
            point_weights=support_point_w,
        )
        support_plane = self._support_plane_from_heights(candidate_heights)
        support_plane = torch.where(
            _pose_ok(candidate_xy_w, foot_yaw),
            support_plane,
            torch.full_like(support_plane, _MISS_PLANE_Z),
        )
        num_pairs, num_candidates, _ = candidate_xy_w.shape
        num_points = local_volume_points.shape[0]
        if num_pairs == 0 or num_points == 0 or not virtual_obstacles:
            return torch.zeros(num_pairs, device=self.device)

        valid_plane = _is_valid_plane(support_plane) & torch.isfinite(
            candidate_xy_w
        ).all(dim=-1)
        ankle_z = torch.where(
            valid_plane, support_plane - volume_z_min, torch.zeros_like(support_plane)
        )
        foot_yaw = _finite_or_zero(foot_yaw)
        xy = _finite_or_zero(candidate_xy_w)

        # foot_yaw: [P] → offsets [P, N]; xy/ankle: [P, K] → points [P, K, N, 3]
        cos_foot = torch.cos(foot_yaw).unsqueeze(-1)
        sin_foot = torch.sin(foot_yaw).unsqueeze(-1)
        lx, ly, lz = local_volume_points.unbind(dim=-1)
        offset_x = cos_foot * lx - sin_foot * ly
        offset_y = sin_foot * lx + cos_foot * ly
        points_w = torch.stack(
            (
                xy[..., 0, None] + offset_x[:, None, :],
                xy[..., 1, None] + offset_y[:, None, :],
                ankle_z[..., None] + lz,
            ),
            dim=-1,
        )
        points_w = _masked_zero(points_w, valid_plane)

        flat_points = points_w.reshape(-1, 3)
        pen_depth = torch.zeros(flat_points.shape[0], device=self.device)
        for obstacle in virtual_obstacles.values():
            offset = _finite_or_zero(obstacle.get_points_penetration_offset(flat_points))
            pen_depth = torch.maximum(pen_depth, torch.norm(offset, dim=-1))
        pen_depth = pen_depth.view(num_pairs, num_candidates, num_points)
        pen_depth = pen_depth * (pen_depth > tolerance).to(pen_depth.dtype)
        pen_depth = _masked_zero(pen_depth, valid_plane)

        if enable_terrain_foot_weights:
            point_w = self._terrain_foot_point_weights(
                local_volume_points,
                volume_x_min,
                volume_x_max,
                family_ids[env_ids],
                stairs_weight_min,
                stairs_weight_max,
            )
            pen_depth = pen_depth * point_w[:, None, :]

        return _finite_or_zero(
            torch.sum(candidate_weights * pen_depth.sum(dim=-1), dim=-1)
        )
