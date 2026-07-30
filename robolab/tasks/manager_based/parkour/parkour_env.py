# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour AMP environment wiring :class:`~robolab.tasks.manager_based.parkour.managers.MultiRewardCfg` to MultiReward."""

from __future__ import annotations

import torch

from isaaclab.envs import VecEnvStepReturn

from robolab.tasks.manager_based.amp.amp_env import AmpEnv
from robolab.tasks.manager_based.parkour.managers import (
    DummyRewardCfg,
    MultiRewardCfg,
    MultiRewardManager,
)
from robolab.tasks.manager_based.parkour.mdp.foothold_imagination import (
    FootholdImaginationManager,
)


def _reward_dict_to_vector(rew: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack multi-group rewards for RSL-RL, which expects a tensor (usually one column)."""
    tensors = tuple(rew.values())
    if not tensors:
        raise ValueError("MultiRewardManager produced an empty reward dict.")
    # Single group 鈫?(num_envs,) as before; multiple groups 鈫?(num_envs, num_groups)
    out = torch.stack(tensors, dim=-1)
    return out.squeeze(-1) if out.shape[-1] == 1 else out


class ParkourEnv(AmpEnv):
    """Same as :class:`~robolab.tasks.manager_based.amp.amp_env.AmpEnv` but swaps in
    :class:`~robolab.tasks.manager_based.parkour.managers.MultiRewardManager` when ``cfg.rewards``
    is a :class:`~robolab.tasks.manager_based.parkour.managers.MultiRewardCfg`.

    Enables curriculum helpers that adjust per-environment reward weights via
    ``get_per_env_term_weights`` / ``set_term_weight_for_envs``.

    RSL-RL expects tensor rewards from ``step``; ``MultiRewardManager.compute`` returns a dict, so we
    convert it here without touching ``rsl_rl``.
    """

    def load_managers(self):
        reward_group_cfg = None
        if isinstance(self.cfg.rewards, MultiRewardCfg):
            reward_group_cfg = self.cfg.rewards
            self.cfg.rewards = DummyRewardCfg()

        super().load_managers()

        # The training runner configures this optional manager from agent cfg.
        self.foothold_guidance = None

        if reward_group_cfg is not None:
            self.cfg.rewards = reward_group_cfg
            self.reward_manager = MultiRewardManager(self.cfg.rewards, self)
            print("[INFO] Multi Reward Manager: ", self.reward_manager)

    def configure_foothold_imagination(self, predictor_cfg: dict) -> None:
        """Create the training-only predictor from the agent configuration."""
        if self.foothold_guidance is not None:
            raise RuntimeError("Foothold imagination has already been configured.")
        self.foothold_guidance = FootholdImaginationManager(
            self, predictor_cfg, self.cfg.foothold_support
        )
        print("[INFO] SSR imagined foothold guidance enabled.")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs, rew, terminated, truncated, extras = super().step(action)
        if self.foothold_guidance is not None:
            reset_env_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            self.foothold_guidance.reset(reset_env_ids)
        if isinstance(rew, dict):
            rew = _reward_dict_to_vector(rew)
            self.reward_buf = rew
        return obs, rew, terminated, truncated, extras

    def prepare_foothold_prediction_step(
        self, privileged_state: torch.Tensor, action: torch.Tensor
    ) -> None:
        """Capture the SSR privileged state and current action before simulation advances."""
        if self.foothold_guidance is not None:
            self.foothold_guidance.prepare_step(privileged_state, action)

    def update_foothold_predictor(self) -> dict[str, float]:
        if self.foothold_guidance is None:
            return {}
        return self.foothold_guidance.update_predictor()
