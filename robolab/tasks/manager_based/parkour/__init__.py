import gymnasium as gym

from . import agents

gym.register(
    id="Atom01-Parkour",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_parkour_env_cfg:Atom01ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_parkour_agent_cfg:Atom01ParkourAmpRunnerCfg",
    },
)

gym.register(
    id="Atom01-Parkour-Play",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_parkour_env_cfg:Atom01ParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.atom01_parkour_agent_cfg:Atom01ParkourAmpRunnerCfg",
    },
)
