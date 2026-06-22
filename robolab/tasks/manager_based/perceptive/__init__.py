"""Perceptive terrain imitation environments."""

import gymnasium as gym

from . import agents

gym.register(
    id="RPO-Perceptive",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_perceptive_env_cfg:RPOPerceptiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_perceptive_agent_cfg:RPOPerceptiveRunnerCfg",
    },
)
