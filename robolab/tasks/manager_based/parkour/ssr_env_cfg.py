# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared SSR environment configuration helpers."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from robolab.tasks.manager_based.parkour.managers import MultiRewardCfg
from robolab.tasks.manager_based.parkour.mdp.foothold_imagination import (
    imagined_foothold_guidance,
)
from robolab.tasks.manager_based.parkour.mdp.foothold_prediction import FootholdSupportCfg
from robolab.tasks.manager_based.parkour.parkour_env_cfg import (
    ParkourEnvCfg,
    ParkourRewardsCfg,
)


@configclass
class FootholdRewardsCfg(MultiRewardCfg):
    """Foothold reward group for its own critic head."""

    imagined_foothold_guidance = RewTerm(
        func=imagined_foothold_guidance,
        weight=-1.0,
        params={
            "enable_terrain_foot_weights": True,
            "stairs_weight_min": 0.0,
            "stairs_weight_max": 1.0,
        },
    )


@configclass
class SSRRewardsCfg(MultiRewardCfg):
    """Task reward groups: locomotion + foothold (style comes from AMP)."""

    locomotion: ParkourRewardsCfg = ParkourRewardsCfg()
    foothold: FootholdRewardsCfg = FootholdRewardsCfg()


def enable_ssr_foothold_guidance(cfg: ParkourEnvCfg) -> None:
    """Attach the foothold reward head and support geometry."""
    locomotion = cfg.rewards.locomotion
    cfg.rewards = SSRRewardsCfg(locomotion=locomotion, foothold=FootholdRewardsCfg())
    cfg.foothold_support = FootholdSupportCfg(
        height_offset=0.035,
        height_tolerance=0.03,
        support_transition_width=0.005,
    )
