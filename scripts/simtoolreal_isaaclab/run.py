"""Launch SimToolReal environment with random actions."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SimToolReal random agent")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
from pathlib import Path

import gymnasium as gym
import torch

# Ensure the project root is on sys.path so our package is importable
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import scripts.simtoolreal_isaaclab  # noqa: F401 — triggers gym.register


def main() -> None:
    from scripts.simtoolreal_isaaclab.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("SimToolReal-Direct-v0", cfg=cfg, disable_env_checker=True)

    print(f"[INFO] obs space: {env.observation_space}")
    print(f"[INFO] act space: {env.action_space}")

    env.reset()
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            obs, reward, terminated, truncated, info = env.step(actions)
            step += 1
            if step % 100 == 0:
                print(f"[Step {step}] reward mean={reward.mean().item():.3f}")

    env.close()


if __name__ == "__main__":
    main()
