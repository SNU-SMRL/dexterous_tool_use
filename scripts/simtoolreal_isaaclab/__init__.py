from __future__ import annotations

import gymnasium as gym

from .simtoolreal_env import SimToolRealEnv
from .simtoolreal_env_cfg import SimToolRealEnvCfg

gym.register(
    id="SimToolReal-Direct-v0",
    entry_point="scripts.simtoolreal_isaaclab:SimToolRealEnv",
    kwargs={"env_cfg_entry_point": SimToolRealEnvCfg},
)
