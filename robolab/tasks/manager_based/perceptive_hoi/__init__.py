
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="RPO-PerceptiveHoi",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_perceptive_hoi_env_cfg:RPOPerceptiveHoiEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_perceptive_hoi_agent_cfg:RPOPerceptiveHoiPPORunnerCfg",
    },
)
