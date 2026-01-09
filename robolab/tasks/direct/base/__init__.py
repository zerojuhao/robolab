from . import agents
from .base_config import BaseAgentCfg, BaseEnvCfg, RewardCfg, HeightScannerCfg, SceneContextCfg, RobotCfg, ObsScalesCfg, NormalizationCfg, CommandRangesCfg, CommandsCfg, NoiseScalesCfg, NoiseCfg, EventCfg
from .base_env import BaseEnv
from .scene_cfg import SceneCfg
from .terrain_generator_cfg import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG, ROUGH_HARD_TERRAINS_CFG
from . import mdp

import gymnasium as gym

gym.register(
    id="Atom01-Flat",
    entry_point=f"{__name__}.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_env_cfg:ATOM01FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_agent_cfg:ATOM01FlatAgentCfg",
    },
)

gym.register(
    id="Atom01-Rough",
    entry_point=f"{__name__}.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_env_cfg:ATOM01RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_agent_cfg:ATOM01RoughAgentCfg",
    },
)