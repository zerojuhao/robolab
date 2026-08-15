# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-force metrics shared by parkour rewards and foothold labels."""

from __future__ import annotations

import torch


def horizontal_force_excess(
    forces_w: torch.Tensor,
    horizontal_ratio: float,
    force_threshold: float,
) -> torch.Tensor:
    """Horizontal force above the scaled vertical force and noise floor."""
    force_xy = torch.linalg.norm(forces_w[..., :2], dim=-1)
    force_z = torch.abs(forces_w[..., 2])
    return torch.relu(force_xy - horizontal_ratio * force_z - force_threshold)


def vertical_contact_mask(
    forces_w: torch.Tensor,
    vertical_force_ratio: float,
    vertical_force_min: float,
) -> torch.Tensor:
    """Whether contact is strong enough and dominated by vertical force."""
    force_xy = torch.linalg.norm(forces_w[..., :2], dim=-1)
    force_z = torch.abs(forces_w[..., 2])
    return (force_z >= vertical_force_min) & (
        force_z > vertical_force_ratio * force_xy
    )
