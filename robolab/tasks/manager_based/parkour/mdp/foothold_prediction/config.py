"""Agent-owned configuration schema for foothold prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class FootholdGridCfg:
    """Gaussian foothold distribution bounds and XY quadrature."""

    sigma_min: float = 0.01
    """Lower bound of isotropic XY sigma, in meters."""
    sigma_max: float = 0.25
    """Upper bound of isotropic XY sigma, in meters."""
    yaw_sigma_min: float = 0.05
    """Lower bound of landing-yaw sigma, in radians."""
    yaw_sigma_max: float = 0.5
    """Upper bound of landing-yaw sigma, in radians."""
    expectation_grid_size: int = 5
    """Odd number of XY quadrature samples per axis."""
    expectation_std_range: float = 2.0
    """XY quadrature half-width in units of sigma."""
    expectation_eval_chunk_size: int = 64
    """Environment-foot pairs per support-evaluation chunk."""


@configclass
class FootholdPredictorCfg:
    """Privileged foothold teacher / guidance training options."""

    enabled: bool = False
    hidden_dims: list[int] = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    ema_decay: float = MISSING
    grid: FootholdGridCfg = MISSING
    nll_loss_coef: float = 1.0
    max_pending_steps: int = MISSING
    pending_sample_stride: int = MISSING
    train_pending_tail_steps: int = MISSING
    """Keep the last K swing steps when labeling a touchdown. ``<=0`` keeps all."""
    train_buffer_capacity: int = MISSING
    batch_size: int = MISSING
    updates_per_iteration: int = MISSING
    min_train_samples: int = MISSING
    curriculum_level_threshold: float = MISSING
    enable_xy_rmse_threshold: float = MISSING
    """Enable swing guidance after teacher XY RMSE falls below this value, in meters."""
    enable_yaw_rmse_threshold: float = MISSING
    """Enable swing guidance after teacher yaw RMSE falls below this value, in radians."""
    clear_train_buffer_on_resume: bool = True


_LEGACY_GRID_KEYS = (
    "resolution",
    "reach_center",
    "reach_radii",
    "reach_x_min",
    "reward_quality_top_k",
    "reward_quality_eval_chunk_size",
    "reward_unselected_mass_penalty",
    "reward_quality_use_forward_window",
    "reward_quality_forward_window_rear_m",
    "reward_quality_forward_window_forward_m",
    "residual_span_cells",
)
_LEGACY_PREDICTOR_KEYS = (
    "classification_loss_coef",
    "residual_loss_coef",
    "residual_neighbor_weight",
    "ce_neighbor_weight",
    "classify_in_reach_only",
    "regress_in_reach_only",
    "regression_loss_coef",
    "input_term_names",
    "input_history_length",
    "input_use_latest_frame",
    "share_with_critic",
)


def normalize_foothold_grid_cfg(cfg: FootholdGridCfg | Mapping) -> FootholdGridCfg:
    if isinstance(cfg, FootholdGridCfg):
        return cfg
    if isinstance(cfg, Mapping):
        values = dict(cfg)
        for key in _LEGACY_GRID_KEYS:
            values.pop(key, None)
        return FootholdGridCfg(**values)
    raise TypeError(f"Unsupported foothold grid config type: {type(cfg)!r}")


def normalize_foothold_predictor_cfg(
    cfg: FootholdPredictorCfg | Mapping,
) -> FootholdPredictorCfg:
    if isinstance(cfg, FootholdPredictorCfg):
        return cfg
    if isinstance(cfg, Mapping):
        values = dict(cfg)
        values["grid"] = normalize_foothold_grid_cfg(values["grid"])
        for key in _LEGACY_PREDICTOR_KEYS:
            values.pop(key, None)
        return FootholdPredictorCfg(**values)
    raise TypeError(f"Unsupported foothold predictor config type: {type(cfg)!r}")
