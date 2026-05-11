"""Play a pretrained SimToolReal rl_games checkpoint in Isaac Lab.

Follows SimToolReal's deployment/rl_player.py pattern:
- Config loaded via OmegaConf (resolves interpolations)
- Obs gets extra dim appended (SAPG coef_cond hack: constant 50.0)
- Uses SimToolReal's forked rl_games with extra_param support
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play pretrained SimToolReal policy")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--checkpoint", type=str, default="simtoolreal/pretrained_policy/model.pth")
parser.add_argument("--config", type=str, default="simtoolreal/pretrained_policy/config.yaml")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml
from omegaconf import OmegaConf
from rl_games.torch_runner import Runner

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# SimToolReal's omegaconf_to_dict
sys.path.insert(0, str(Path(_project_root) / "simtoolreal"))
from isaacgymenvs.utils.reformat import omegaconf_to_dict

import scripts.simtoolreal_isaaclab  # noqa: F401

from scripts.simtoolreal_isaaclab.convention_wrapper import IsaacGymConventionWrapper


def load_simtoolreal_config(config_path: str, device: str) -> dict:
    """Load config following SimToolReal's read_cfg pattern."""
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = omegaconf_to_dict(OmegaConf.create(raw_cfg))
    if "train" in cfg and "params" in cfg["train"]:
        cfg["train"]["params"]["config"]["device"] = device
        cfg["train"]["params"]["config"]["device_name"] = device
    return cfg


def main() -> None:
    from scripts.simtoolreal_isaaclab.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("SimToolReal-Direct-v0", cfg=cfg, disable_env_checker=True)
    env = IsaacGymConventionWrapper(env)

    device = "cuda:0"

    # Load config (SimToolReal pattern: cfg["train"] contains rl_games config)
    full_cfg = load_simtoolreal_config(args_cli.config, device)
    train_cfg = full_cfg["train"]

    # Register env with rl_games (SimToolReal pattern: env is self-registering)
    from rl_games.common import env_configurations

    env_configurations.register(
        "rlgpu", {"env_creator": lambda **kwargs: env, "vecenv_type": "RLGPU"}
    )

    # Load checkpoint
    train_cfg["load_path"] = args_cli.checkpoint
    train_cfg["params"]["config"]["num_actors"] = args_cli.num_envs

    runner = Runner()
    runner.load(train_cfg)
    player = runner.create_player()
    player.init_rnn()
    player.has_batch_dimension = True
    player.restore(args_cli.checkpoint)

    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Network: {player.model}")

    # Play loop — wrapper returns old gym interface (obs, reward, done, info)
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["policy"]

    step = 0
    max_successes = env.unwrapped.max_consecutive_successes
    ep_goal_pcts: list[float] = []

    while simulation_app.is_running():
        with torch.inference_mode():
            obs_with_id = torch.cat(
                [obs, 50.0 + torch.zeros(args_cli.num_envs, 1, device=device)],
                dim=-1,
            )
            actions = player.get_action(obs_with_id, is_deterministic=True)
            actions = actions.reshape(-1, 29)

            obs_raw, reward, dones, info = env.step(actions)
            if isinstance(obs_raw, dict):
                obs = obs_raw["policy"]
            else:
                obs = obs_raw

            step += 1

            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                for did in done_ids:
                    goal_pct = 100.0 * env.unwrapped.successes[did].item() / max_successes
                    ep_goal_pcts.append(goal_pct)
                if player.is_rnn and player.states is not None:
                    for s in player.states:
                        s[:, done_ids, :] = 0.0

            if step % 200 == 0 and ep_goal_pcts:
                recent = ep_goal_pcts[-20:]
                avg = sum(recent) / len(recent)
                print(f"[Step {step}] episodes={len(ep_goal_pcts)}, "
                      f"avg goal%={avg:.1f}% "
                      f"(max={max(recent):.1f}%)")

    env.close()


if __name__ == "__main__":
    main()
