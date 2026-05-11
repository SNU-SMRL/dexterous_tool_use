"""SimToolReal environment for Isaac Lab.

Phase 1: Scene setup + skeleton stubs for all abstract methods.
Phase 2: Reset + action logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_apply,
    sample_uniform,
    saturate,
    scale_transform,
    unscale_transform,
)

if TYPE_CHECKING:
    from .simtoolreal_env_cfg import SimToolRealEnvCfg


class SimToolRealEnv(DirectRLEnv):
    cfg: SimToolRealEnvCfg

    def __init__(
        self,
        cfg: SimToolRealEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        # --- Joint indices ---
        self.num_arm_dofs = 7
        self.num_hand_dofs = self.robot.num_joints - self.num_arm_dofs
        self.num_dofs = self.robot.num_joints

        # --- Joint limits ---
        dof_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower = dof_limits[..., 0]
        self.dof_upper = dof_limits[..., 1]

        # --- Body indices ---
        from .simtoolreal_env_cfg import (
            FINGERTIP_BODY_NAMES,
            FINGERTIP_OFFSETS,
            PALM_BODY_NAME,
            PALM_CENTER_OFFSET,
        )

        self.palm_body_idx = self.robot.body_names.index(PALM_BODY_NAME)
        self.fingertip_body_ids = [
            self.robot.body_names.index(name) for name in FINGERTIP_BODY_NAMES
        ]
        self.palm_center_offset = torch.tensor(
            PALM_CENTER_OFFSET, dtype=torch.float32, device=self.device
        )
        self.fingertip_offsets = torch.tensor(
            FINGERTIP_OFFSETS, dtype=torch.float32, device=self.device
        )

        # --- Control buffers ---
        self.prev_targets = torch.zeros(
            self.num_envs, self.num_dofs, dtype=torch.float32, device=self.device
        )
        self.cur_targets = torch.zeros_like(self.prev_targets)

        # --- Goal buffers ---
        self.goal_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.goal_quat = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        self.goal_quat[:, 0] = 1.0  # wxyz identity

        # --- Keypoint offsets (4 corners of unit cube, scaled by object size) ---
        from .simtoolreal_env_cfg import KEYPOINT_OFFSETS

        kp_raw = torch.tensor(KEYPOINT_OFFSETS, dtype=torch.float32, device=self.device)
        kp_scale = self.cfg.object_base_size * self.cfg.keypoint_scale / 2.0
        self.keypoint_offsets = kp_raw * kp_scale  # (4, 3)
        self.num_keypoints = len(KEYPOINT_OFFSETS)
        self.num_fingertips = len(self.fingertip_body_ids)

        # --- Object scale buffer (Phase 1: fixed, Phase 2.5: per-env) ---
        self.object_scales = torch.ones(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )

        # --- Reward tracking ---
        self.lifted_object = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.near_goal_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.goal_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Best-so-far distances (initialized to -1, set on first frame)
        self.closest_fingertip_dist = -torch.ones(
            self.num_envs, self.num_fingertips, dtype=torch.float32, device=self.device
        )
        self.closest_keypoint_max_dist = -torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.object_init_z = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # Compute observation space size
        self._obs_dims = {
            "joint_pos": self.num_dofs,          # 29
            "joint_vel": self.num_dofs,          # 29
            "prev_action_targets": self.num_dofs,  # 29
            "palm_pos": 3,
            "palm_rot": 4,
            "object_rot": 4,
            "fingertip_pos_rel_palm": self.num_fingertips * 3,  # 15
            "keypoints_rel_palm": self.num_keypoints * 3,  # 12
            "keypoints_rel_goal": self.num_keypoints * 3,  # 12
            "object_scales": 3,
        }
        computed_obs_size = sum(self._obs_dims.values())
        print(f"[SimToolRealEnv] Observation size: {computed_obs_size} "
              f"(cfg: {self.cfg.observation_space})")

        print(f"[SimToolRealEnv] {self.num_dofs} joints, obs={computed_obs_size}")

    # ---------------------------------------------------------------
    # Scene setup
    # ---------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Table: spawn as static scene geometry (not tracked by RigidObject)
        self.cfg.table_cfg.func(
            "/World/envs/env_.*/Table",
            self.cfg.table_cfg,
            translation=self.cfg.table_pos,
        )

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # --- Robot joints: default + noise ---
        default_pos = self.robot.data.default_joint_pos[env_ids]
        delta_max = self.dof_upper[env_ids] - default_pos
        delta_min = self.dof_lower[env_ids] - default_pos
        rand_delta = delta_min + (delta_max - delta_min) * torch.rand_like(default_pos)

        noise_coeff = torch.ones(
            num_reset, self.num_dofs, dtype=torch.float32, device=self.device
        )
        noise_coeff[:, : self.num_arm_dofs] = self.cfg.reset_dof_pos_noise_arm
        noise_coeff[:, self.num_arm_dofs :] = self.cfg.reset_dof_pos_noise_fingers

        joint_pos = default_pos + noise_coeff * rand_delta
        joint_pos = torch.clamp(joint_pos, self.dof_lower[env_ids], self.dof_upper[env_ids])

        joint_vel = (
            self.cfg.reset_dof_vel_noise
            * sample_uniform(-1.0, 1.0, (num_reset, self.num_dofs), device=self.device)
        )

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        self.prev_targets[env_ids] = joint_pos
        self.cur_targets[env_ids] = joint_pos

        # --- Object: table top + noise ---
        object_pose = self.object.data.default_root_state[env_ids, :7].clone()
        object_pose[:, 0] += self.cfg.reset_position_noise_x * (
            2.0 * torch.rand(num_reset, device=self.device) - 1.0
        )
        object_pose[:, 1] += self.cfg.reset_position_noise_y * (
            2.0 * torch.rand(num_reset, device=self.device) - 1.0
        )
        object_pose[:, 2] += self.cfg.reset_position_noise_z * (
            2.0 * torch.rand(num_reset, device=self.device) - 1.0
        )
        object_pose[:, :3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(object_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros(num_reset, 6, device=self.device), env_ids=env_ids
        )

        # --- Goal ---
        self._resample_goal(env_ids)

        # --- Reward buffers ---
        self.lifted_object[env_ids] = False
        self.near_goal_steps[env_ids] = 0
        self.goal_reached[env_ids] = False
        self.closest_fingertip_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.object_init_z[env_ids] = object_pose[:, 2] - self.scene.env_origins[env_ids, 2]

    # ---------------------------------------------------------------
    # Action
    # ---------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        dt = self.cfg.sim.dt * self.cfg.decimation

        # Arm: relative incremental control
        arm_targets = (
            self.robot.data.joint_pos[:, : self.num_arm_dofs]
            + self.cfg.dof_speed_scale * dt * self.actions[:, : self.num_arm_dofs]
        )
        arm_targets = (
            self.cfg.arm_moving_average * arm_targets
            + (1.0 - self.cfg.arm_moving_average) * self.prev_targets[:, : self.num_arm_dofs]
        )

        # Hand: [-1, 1] → joint limits, with smoothing
        hand_targets = scale_transform(
            self.actions[:, self.num_arm_dofs :],
            self.dof_lower[:, self.num_arm_dofs :],
            self.dof_upper[:, self.num_arm_dofs :],
        )
        hand_targets = (
            self.cfg.hand_moving_average * hand_targets
            + (1.0 - self.cfg.hand_moving_average) * self.prev_targets[:, self.num_arm_dofs :]
        )

        targets = torch.cat([arm_targets, hand_targets], dim=-1)
        targets = saturate(targets, self.dof_lower, self.dof_upper)

        self.prev_targets[:] = self.cur_targets
        self.cur_targets[:] = targets
        self.robot.set_joint_position_target(targets)

    # ---------------------------------------------------------------
    # Observations (Phase 3)
    # ---------------------------------------------------------------
    def _compute_intermediate_values(self) -> None:
        """Cache frequently used tensors from sim state."""
        env_origins = self.scene.env_origins

        # Joint state
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Palm (iiwa14_link_7) — local frame
        palm_pos_w = self.robot.data.body_pos_w[:, self.palm_body_idx]
        self.palm_quat = self.robot.data.body_quat_w[:, self.palm_body_idx]  # wxyz
        self.palm_pos = palm_pos_w - env_origins
        # Apply palm center offset in palm's local frame
        palm_offset_batch = self.palm_center_offset.unsqueeze(0).expand(self.num_envs, -1)
        self.palm_center = self.palm_pos + quat_apply(self.palm_quat, palm_offset_batch)

        self.palm_vel = self.robot.data.body_vel_w[:, self.palm_body_idx]  # (N, 6)

        # Fingertips — local frame with offsets
        ft_pos_w = self.robot.data.body_pos_w[:, self.fingertip_body_ids]  # (N, 5, 3)
        ft_quat_w = self.robot.data.body_quat_w[:, self.fingertip_body_ids]  # (N, 5, 4)
        ft_offsets_rotated = quat_apply(
            ft_quat_w.reshape(-1, 4),
            self.fingertip_offsets.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3),
        ).reshape(self.num_envs, self.num_fingertips, 3)
        self.fingertip_pos = ft_pos_w + ft_offsets_rotated - env_origins.unsqueeze(1)

        # Object — local frame
        self.object_pos = self.object.data.root_pos_w - env_origins
        self.object_quat = self.object.data.root_quat_w  # wxyz
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # Keypoints — 4 corners in world frame, then local
        kp_offsets_expanded = self.keypoint_offsets.unsqueeze(0).expand(
            self.num_envs, -1, -1
        )  # (N, 4, 3)
        obj_quat_expanded = self.object_quat.unsqueeze(1).expand(
            -1, self.num_keypoints, -1
        )  # (N, 4, 4)
        self.keypoints = self.object_pos.unsqueeze(1) + quat_apply(
            obj_quat_expanded.reshape(-1, 4),
            kp_offsets_expanded.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_keypoints, 3)

        # Goal keypoints
        goal_quat_expanded = self.goal_quat.unsqueeze(1).expand(
            -1, self.num_keypoints, -1
        )
        self.goal_keypoints = self.goal_pos.unsqueeze(1) + quat_apply(
            goal_quat_expanded.reshape(-1, 4),
            kp_offsets_expanded.reshape(-1, 3),
        ).reshape(self.num_envs, self.num_keypoints, 3)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        obs = torch.cat(
            [
                # Joint state (scaled to [-1, 1])
                unscale_transform(self.joint_pos, self.dof_lower, self.dof_upper),
                self.joint_vel,
                self.prev_targets,
                # Palm
                self.palm_center,
                self.palm_quat,
                # Object
                self.object_quat,
                # Fingertips relative to palm
                (self.fingertip_pos - self.palm_center.unsqueeze(1)).reshape(
                    self.num_envs, -1
                ),
                # Keypoints relative to palm
                (self.keypoints - self.palm_center.unsqueeze(1)).reshape(
                    self.num_envs, -1
                ),
                # Keypoints relative to goal
                (self.keypoints - self.goal_keypoints).reshape(self.num_envs, -1),
                # Object scales
                self.object_scales,
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ---------------------------------------------------------------
    # Rewards (Phase 4)
    # ---------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # --- Fingertip distances to object ---
        ft_to_obj = self.fingertip_pos - self.object_pos.unsqueeze(1)  # (N, 5, 3)
        curr_ft_dist = torch.norm(ft_to_obj, dim=-1)  # (N, 5)

        # Initialize best-so-far on first frame
        first_frame = self.closest_fingertip_dist[:, 0] < 0.0
        if first_frame.any():
            self.closest_fingertip_dist[first_frame] = curr_ft_dist[first_frame]

        # Fingertip delta reward (only before lifting)
        ft_deltas = torch.clamp(self.closest_fingertip_dist - curr_ft_dist, 0, 10)
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, curr_ft_dist)
        fingertip_delta_rew = ft_deltas.sum(dim=-1) * (~self.lifted_object)

        # --- Lifting reward ---
        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_z
        lifting_rew = torch.clamp(z_lift, 0, 0.5)
        lifted_object = (z_lift > self.cfg.lifting_bonus_threshold) | self.lifted_object
        just_lifted = lifted_object & (~self.lifted_object)
        lift_bonus_rew = self.cfg.lifting_bonus * just_lifted.float()
        lifting_rew = lifting_rew * (~lifted_object).float()
        self.lifted_object = lifted_object

        # --- Keypoint reward (only after lifting) ---
        kp_dists = torch.norm(self.keypoints - self.goal_keypoints, dim=-1)  # (N, 4)
        keypoints_max_dist = kp_dists.max(dim=-1).values  # (N,)

        first_kp = self.closest_keypoint_max_dist < 0.0
        if first_kp.any():
            self.closest_keypoint_max_dist[first_kp] = keypoints_max_dist[first_kp]

        kp_deltas = torch.clamp(self.closest_keypoint_max_dist - keypoints_max_dist, 0, 100)
        self.closest_keypoint_max_dist = torch.minimum(
            self.closest_keypoint_max_dist, keypoints_max_dist
        )
        keypoint_rew = kp_deltas * lifted_object.float()

        # --- Action penalties ---
        arm_penalty = (
            -torch.sum(torch.abs(self.joint_vel[:, :self.num_arm_dofs]), dim=-1)
            * self.cfg.arm_actions_penalty_scale
        )
        hand_penalty = (
            -torch.sum(torch.abs(self.joint_vel[:, self.num_arm_dofs:]), dim=-1)
            * self.cfg.hand_actions_penalty_scale
        )

        # --- Goal success ---
        kp_tolerance = self.cfg.success_tolerance * self.cfg.keypoint_scale
        near_goal = keypoints_max_dist <= kp_tolerance
        self.near_goal_steps = (self.near_goal_steps + near_goal.int()) * near_goal.int()
        self.goal_reached = self.near_goal_steps >= self.cfg.success_steps
        bonus_rew = near_goal.float() * (self.cfg.reach_goal_bonus / self.cfg.success_steps)

        # --- Total ---
        reward = (
            fingertip_delta_rew * self.cfg.distance_delta_rew_scale
            + lifting_rew * self.cfg.lifting_rew_scale
            + lift_bonus_rew
            + keypoint_rew * self.cfg.keypoint_rew_scale
            + arm_penalty
            + hand_penalty
            + bonus_rew
        )
        return reward

    # ---------------------------------------------------------------
    # Dones (Phase 4)
    # ---------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        fallen = self.object_pos[:, 2] < 0.1
        palm_obj_dist = torch.norm(self.palm_center - self.object_pos, dim=-1)
        too_far = palm_obj_dist > 1.5
        terminated = fallen | too_far | self.goal_reached

        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    # ---------------------------------------------------------------
    # Goal sampling
    # ---------------------------------------------------------------
    def _resample_goal(self, env_ids: Sequence[int]) -> None:
        num = len(env_ids)
        target_min = torch.tensor(
            self.cfg.target_volume_mins, device=self.device
        )
        target_max = torch.tensor(
            self.cfg.target_volume_maxs, device=self.device
        )
        self.goal_pos[env_ids] = sample_uniform(
            target_min, target_max, (num, 3), device=self.device
        )
        # Identity rotation for now; delta rotation in Phase 4
        self.goal_quat[env_ids, 0] = 1.0
        self.goal_quat[env_ids, 1:] = 0.0
