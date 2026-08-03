"""Agent-owned configuration schema for foothold prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class FootholdGridCfg:
    """Distribution bounds and quadrature for Gaussian foothold predictions.

    The Gaussian mean is an unconstrained base-frame XY value, as in SSR. The
    support expectation is approximated with a square grid in normalized
    standard-deviation coordinates.
    """

    sigma_min: float = 0.01
    sigma_max: float = 0.25
    expectation_grid_size: int = 5
    expectation_std_range: float = 2.0
    expectation_eval_chunk_size: int = 64


@configclass
class FootholdPredictorCfg:
    """Model and training options whose concrete values belong in the agent cfg."""

    enabled: bool = False
    hidden_dims: list[int] = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    ema_decay: float = MISSING
    grid: FootholdGridCfg = MISSING
    nll_loss_coef: float = 1.0
    max_pending_steps: int = MISSING
    pending_sample_stride: int = MISSING
    # Keep only the last K pending swing steps when labeling touchdown (≤0 => keep all).
    train_pending_tail_steps: int = MISSING
    train_buffer_capacity: int = MISSING
    batch_size: int = MISSING
    updates_per_iteration: int = MISSING
    min_train_samples: int = MISSING
    curriculum_level_threshold: float = MISSING
    # Enable guidance after the Gaussian mean reaches this XY RMSE.
    enable_xy_rmse_threshold: float = MISSING
    # Drop replay samples when loading a checkpoint (avoids stale terrain labels).
    clear_train_buffer_on_resume: bool = True


def normalize_foothold_grid_cfg(cfg: FootholdGridCfg | Mapping) -> FootholdGridCfg:
    if isinstance(cfg, FootholdGridCfg):
        return cfg
    if isinstance(cfg, Mapping):
        values = dict(cfg)
        # Drop legacy categorical-grid / point-regression keys.
        for legacy in (
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
        ):
            values.pop(legacy, None)
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
        for legacy in (
            "classification_loss_coef",
            "residual_loss_coef",
            "residual_neighbor_weight",
            "ce_neighbor_weight",
            "classify_in_reach_only",
            "regress_in_reach_only",
            "regression_loss_coef",
        ):
            values.pop(legacy, None)
        return FootholdPredictorCfg(**values)
    raise TypeError(f"Unsupported foothold predictor config type: {type(cfg)!r}")
