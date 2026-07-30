"""Reachable base-frame foothold probability grid."""

from __future__ import annotations

import torch


class ReachableFootholdGrid:
    """Build a mirrored per-foot reachable mask over a shared XY grid."""

    def __init__(self, cfg, device: torch.device | str) -> None:
        self.cfg = cfg
        self.device = device
        self._validate_cfg()

        self.x_values = self._axis_values(cfg.x_range, cfg.resolution)
        self.y_values = self._axis_values(cfg.y_range, cfg.resolution)
        self.num_x = int(self.x_values.numel())
        self.num_y = int(self.y_values.numel())
        self.points = torch.cartesian_prod(self.x_values, self.y_values)
        self.num_cells = self.points.shape[0]

        center_x, center_y = cfg.reach_center
        radius_x, radius_y = cfg.reach_radii
        self.reach_centers = torch.tensor(
            ((center_x, center_y), (center_x, -center_y)),
            dtype=torch.float32,
            device=device,
        )
        self.reach_radii = torch.tensor(
            (radius_x, radius_y), dtype=torch.float32, device=device
        )
        normalized_offsets = (
            self.points.unsqueeze(0) - self.reach_centers.unsqueeze(1)
        ) / self.reach_radii
        self.reachable_mask = normalized_offsets.square().sum(dim=-1) <= 1.0

        reachable_counts = self.reachable_mask.sum(dim=-1)
        min_reachable = int(reachable_counts.min().item())
        if (reachable_counts == 0).any():
            raise ValueError("Each foot must have at least one reachable grid cell.")
        quality_top_k = int(getattr(cfg, "reward_quality_top_k", 64))
        if quality_top_k <= 0 or quality_top_k > min_reachable:
            raise ValueError(
                "reward_quality_top_k must be positive and no larger than the "
                f"reachable cell count per foot, got {quality_top_k}."
            )
        quality_chunk_size = int(
            getattr(cfg, "reward_quality_eval_chunk_size", 64)
        )
        if quality_chunk_size <= 0:
            raise ValueError("reward_quality_eval_chunk_size must be positive.")

        unselected_mass_penalty = float(
            getattr(cfg, "reward_unselected_mass_penalty", 1.0)
        )
        if not (0.0 <= unselected_mass_penalty <= 1.0):
            raise ValueError(
                "reward_unselected_mass_penalty must be in [0, 1], "
                f"got {unselected_mass_penalty}."
            )

        self._build_neighbor_tables()

    def _validate_cfg(self) -> None:
        if self.cfg.resolution <= 0.0:
            raise ValueError("Foothold grid resolution must be positive.")
        if self.cfg.x_range[0] >= self.cfg.x_range[1]:
            raise ValueError("Foothold grid x_range must be increasing.")
        if self.cfg.y_range[0] >= self.cfg.y_range[1]:
            raise ValueError("Foothold grid y_range must be increasing.")
        if min(self.cfg.reach_radii) <= 0.0:
            raise ValueError("Foothold reach radii must be positive.")
        residual_span = float(getattr(self.cfg, "residual_span_cells", 0.5))
        if residual_span <= 0.0:
            raise ValueError("Foothold residual_span_cells must be positive.")

    def _axis_values(
        self, value_range: tuple[float, float], resolution: float
    ) -> torch.Tensor:
        span_in_cells = (value_range[1] - value_range[0]) / resolution
        num_intervals = round(span_in_cells)
        if abs(span_in_cells - num_intervals) > 1.0e-5:
            raise ValueError(
                f"Grid range {value_range} must be divisible by resolution {resolution}."
            )
        return torch.linspace(
            value_range[0],
            value_range[1],
            num_intervals + 1,
            device=self.device,
        )

    def _build_neighbor_tables(self) -> None:
        """Precompute self + 4-connected neighbor indices for residual supervision."""
        # cartesian_prod(x, y) lays out as ix-major: index = ix * num_y + iy.
        cell_ids = torch.arange(self.num_cells, device=self.device)
        ix = cell_ids // self.num_y
        iy = cell_ids % self.num_y
        offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        neighbor_indices = []
        neighbor_valid = []
        for dx, dy in offsets:
            nix = ix + dx
            niy = iy + dy
            valid = (nix >= 0) & (nix < self.num_x) & (niy >= 0) & (niy < self.num_y)
            indices = (nix * self.num_y + niy).clamp(min=0, max=self.num_cells - 1)
            neighbor_indices.append(indices)
            neighbor_valid.append(valid)
        # [num_cells, 5]
        self.neighbor_indices = torch.stack(neighbor_indices, dim=-1)
        self.neighbor_valid = torch.stack(neighbor_valid, dim=-1)

    def masked_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Set unreachable grid logits to the lowest representable value."""
        if logits.shape[-2:] != (2, self.num_cells):
            raise ValueError(
                "Foothold logits must have trailing shape "
                f"(2, {self.num_cells}), got {tuple(logits.shape[-2:])}."
            )
        return logits.masked_fill(
            ~self.reachable_mask.unsqueeze(0), torch.finfo(logits.dtype).min
        )

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask unreachable cells and normalize each foot's categorical map."""
        return torch.softmax(self.masked_logits(logits), dim=-1)

    @property
    def max_residual(self) -> float:
        """Maximum absolute XY residual in either axis."""
        span = float(getattr(self.cfg, "residual_span_cells", 0.5))
        return span * float(self.cfg.resolution)

    def bounded_residuals(self, raw_residuals: torch.Tensor) -> torch.Tensor:
        """Map unconstrained residual outputs into the configured residual box."""
        return self.max_residual * torch.tanh(raw_residuals)

    def target_indices(
        self, targets: torch.Tensor, foot_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map continuous touchdown XY labels to their nearest reachable cells."""
        distances = (targets[:, None, :] - self.points[None]).square().sum(dim=-1)
        reachable = self.reachable_mask[foot_ids]
        indices = distances.masked_fill(~reachable, torch.inf).argmin(dim=-1)

        centers = self.reach_centers[foot_ids]
        in_ellipse = ((targets - centers) / self.reach_radii).square().sum(
            dim=-1
        ) <= 1.0
        in_bounds = (
            (targets[:, 0] >= self.cfg.x_range[0])
            & (targets[:, 0] <= self.cfg.x_range[1])
            & (targets[:, 1] >= self.cfg.y_range[0])
            & (targets[:, 1] <= self.cfg.y_range[1])
        )
        target_residuals = targets - self.points[indices]
        residual_encodable = (target_residuals.abs() <= self.max_residual + 1.0e-6).all(
            dim=-1
        )
        return indices, in_ellipse & in_bounds & residual_encodable

    def mode_xy(
        self, probabilities: torch.Tensor, residuals: torch.Tensor
    ) -> torch.Tensor:
        """Return the refined coordinate of each sample's most probable cell."""
        indices = probabilities.argmax(dim=-1)
        batch_ids = torch.arange(probabilities.shape[0], device=self.device)
        return self.points[indices] + residuals[batch_ids, indices]

    def cell_xy_indices(self, cell_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode flat cell ids into (ix, iy) on the shared XY lattice."""
        return cell_ids // self.num_y, cell_ids % self.num_y

    @property
    def signature(self) -> tuple:
        """Serializable geometry signature used for checkpoint compatibility."""
        return (
            "categorical_xy_residual_v2",
            tuple(self.cfg.x_range),
            tuple(self.cfg.y_range),
            float(self.cfg.resolution),
            tuple(self.cfg.reach_center),
            tuple(self.cfg.reach_radii),
            float(getattr(self.cfg, "residual_span_cells", 0.5)),
            int(self.num_cells),
        )
