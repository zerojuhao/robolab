"""Agent-owned configuration schema for foothold prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class FootholdGridCfg:
    """Compact reachable-region grid in the robot base frame for both feet.

    Lattice points are generated inside the tight bounding box of the mirrored
    reach ellipses; only cells that fall inside at least one foot ellipse are
    retained.  The network predicts logits over this compact set directly.
    """

    resolution: float = MISSING
    reach_center: tuple[float, float] = MISSING
    reach_radii: tuple[float, float] = MISSING
    # Rank every reachable cell by terrain support quality and retain the best K.
    reward_quality_top_k: int = 64
    # Bound peak raycast memory while still evaluating the full reachable set.
    reward_quality_eval_chunk_size: int = 64
    # Unsupported cost assigned to probability mass outside the quality Top-K.
    reward_unselected_mass_penalty: float = 1.0
    # Residual half-width in units of ``resolution`` (1.0 => ± one cell).
    residual_span_cells: float = 1.0


@configclass
class FootholdPredictorCfg:
    """Model and training options whose concrete values belong in the agent cfg."""

    enabled: bool = False
    hidden_dims: list[int] = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    ema_decay: float = MISSING
    grid: FootholdGridCfg = MISSING
    # L = classification_loss_coef * CE + residual_loss_coef * L_residual
    classification_loss_coef: float = 1.0
    residual_loss_coef: float = MISSING
    # Relative weight for residual loss on 4-neighbor cells (GT cell weight is 1).
    residual_neighbor_weight: float = 0.25
    # Soft CE mass on each 4-neighbor relative to GT (=1). 0 => hard one-hot CE.
    ce_neighbor_weight: float = 0.0
    # If True, CE is averaged only over in-reach touchdown labels.
    classify_in_reach_only: bool = True
    max_pending_steps: int = MISSING
    pending_sample_stride: int = MISSING
    # Keep only the last K pending swing steps when labeling touchdown (≤0 => keep all).
    train_pending_tail_steps: int = MISSING
    train_buffer_capacity: int = MISSING
    batch_size: int = MISSING
    updates_per_iteration: int = MISSING
    min_train_samples: int = MISSING
    curriculum_level_threshold: float = MISSING
    # Enable foothold reward once mode XY RMSE (meters) falls below this.
    enable_xy_rmse_threshold: float = MISSING


def normalize_foothold_grid_cfg(cfg: FootholdGridCfg | Mapping) -> FootholdGridCfg:
    if isinstance(cfg, FootholdGridCfg):
        return cfg
    if isinstance(cfg, Mapping):
        return FootholdGridCfg(**dict(cfg))
    raise TypeError(f"Unsupported foothold grid config type: {type(cfg)!r}")


def normalize_foothold_predictor_cfg(
    cfg: FootholdPredictorCfg | Mapping,
) -> FootholdPredictorCfg:
    if isinstance(cfg, FootholdPredictorCfg):
        return cfg
    if isinstance(cfg, Mapping):
        values = dict(cfg)
        values["grid"] = normalize_foothold_grid_cfg(values["grid"])
        return FootholdPredictorCfg(**values)
    raise TypeError(f"Unsupported foothold predictor config type: {type(cfg)!r}")
