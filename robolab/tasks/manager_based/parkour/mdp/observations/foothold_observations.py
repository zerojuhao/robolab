# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Foothold predictor observation helpers."""

from __future__ import annotations

from typing import Sequence

import torch
from tensordict import TensorDict

FOOTHOLD_EXCLUDE_TERMS: frozenset[str] = frozenset({"foothold_teacher_xy"})


def build_foothold_predictor_state(
    critic_obs: TensorDict,
    exclude_terms: Sequence[str] | None = None,
) -> torch.Tensor:
    """Flatten critic terms into the privileged foothold-teacher input."""
    exclude = set(exclude_terms) if exclude_terms is not None else set(FOOTHOLD_EXCLUDE_TERMS)
    parts = [term for name, term in critic_obs.items() if name not in exclude]
    if not parts:
        raise ValueError(
            "No foothold predictor input terms found in critic obs after filtering. "
            f"exclude={sorted(exclude)}."
        )
    return torch.cat(parts, dim=-1)
