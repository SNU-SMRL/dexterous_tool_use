"""Convention wrapper: Isaac Lab (BFS, wxyz) ↔ IsaacGym (DFS, xyzw).

Wraps a SimToolRealEnv so that a pretrained IsaacGym policy can interact
with the Isaac Lab environment without modification.

Obs flow:  env (BFS, wxyz) → wrapper → policy (DFS, xyzw)
Act flow:  policy (DFS)    → wrapper → env (BFS)
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import torch

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.asset_conversion.joint_remapping import (
    ISAACGYM_TO_ISAACLAB,
    ISAACLAB_TO_ISAACGYM,
    NUM_DOFS,
)

# Observation layout produced by SimToolRealEnv._get_observations()
# All in Isaac Lab convention (BFS joints, wxyz quaternions)
_OBS_SLICES: dict[str, tuple[int, int, str]] = {
    # (start, end, conversion_type)
    "joint_pos":              (0,   29,  "joints"),
    "joint_vel":              (29,  58,  "joints"),
    "prev_action_targets":    (58,  87,  "joints"),
    "palm_pos":               (87,  90,  "none"),
    "palm_rot":               (90,  94,  "quat"),
    "object_rot":             (94,  98,  "quat"),
    "fingertip_pos_rel_palm": (98,  113, "none"),
    "keypoints_rel_palm":     (113, 125, "none"),
    "keypoints_rel_goal":     (125, 137, "none"),
    "object_scales":          (137, 140, "none"),
}


def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from (w,x,y,z) to (x,y,z,w)."""
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from (x,y,z,w) to (w,x,y,z)."""
    return torch.cat([q[..., 3:], q[..., :3]], dim=-1)


class IsaacGymConventionWrapper(gym.Wrapper):
    """Translates between Isaac Lab env convention and IsaacGym policy convention.

    The wrapped env uses BFS joint order and wxyz quaternions (Isaac Lab native).
    This wrapper converts obs to DFS/xyzw for the pretrained policy, and converts
    actions back from DFS to BFS for the env.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        device = self.device

        # rl_games expects 1D spaces (obs_dim,), not (num_envs, obs_dim)
        import numpy as np
        from gymnasium import spaces

        obs_dim = env.unwrapped.cfg.observation_space
        act_dim = env.unwrapped.cfg.action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32,
        )

        # rl_games compatibility (SimToolReal rl_player.py pattern)
        self.set_env_state = lambda *args, **kwargs: None
        self._lab_to_gym = torch.tensor(
            ISAACLAB_TO_ISAACGYM, dtype=torch.long, device=device,
        )
        self._gym_to_lab = torch.tensor(
            ISAACGYM_TO_ISAACLAB, dtype=torch.long, device=device,
        )

    def _convert_obs_to_isaacgym(self, obs: torch.Tensor) -> torch.Tensor:
        """Convert observation from Isaac Lab convention to IsaacGym convention."""
        out = obs.clone()
        for name, (start, end, conv) in _OBS_SLICES.items():
            if conv == "joints":
                out[..., start:end] = obs[..., start:end].index_select(-1, self._lab_to_gym)
            elif conv == "quat":
                out[..., start:end] = _wxyz_to_xyzw(obs[..., start:end])
        return out

    def _convert_action_to_isaaclab(self, action: torch.Tensor) -> torch.Tensor:
        """Convert action from IsaacGym DFS order to Isaac Lab BFS order."""
        return action.index_select(-1, self._gym_to_lab)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # gymnasium returns (obs, info), old gym returns obs
        if isinstance(result, tuple):
            obs = result[0]
        else:
            obs = result
        if isinstance(obs, dict):
            obs["policy"] = self._convert_obs_to_isaacgym(obs["policy"])
        else:
            obs = self._convert_obs_to_isaacgym(obs)
        return obs

    def step(self, action: torch.Tensor):
        action_lab = self._convert_action_to_isaaclab(action)
        result = self.env.step(action_lab)
        # gymnasium returns 5 values, old gym expects 4
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated | truncated
        else:
            obs, reward, done, info = result
        if isinstance(obs, dict):
            obs["policy"] = self._convert_obs_to_isaacgym(obs["policy"])
        else:
            obs = self._convert_obs_to_isaacgym(obs)
        return obs, reward, done, info
