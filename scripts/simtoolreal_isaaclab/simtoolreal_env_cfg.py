"""SimToolReal environment configuration for Isaac Lab.

Ported from SimToolReal (IsaacGymEnvs) to Isaac Lab DirectRLEnv.
All joint-indexed arrays are in Isaac Lab BFS order.
"""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

ASSET_DIR = str(Path(__file__).resolve().parents[2] / "assets" / "usd")

# ---------------------------------------------------------------------------
# Joint names in Isaac Lab BFS order (from scripts/asset_conversion/joint_remapping.py)
# ---------------------------------------------------------------------------
JOINT_NAMES_BFS: list[str] = [
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
    "left_1_thumb_CMC_FE",
    "left_2_index_MCP_FE",
    "left_3_middle_MCP_FE",
    "left_4_ring_MCP_FE",
    "left_5_pinky_CMC",
    "left_thumb_CMC_AA",
    "left_index_MCP_AA",
    "left_middle_MCP_AA",
    "left_ring_MCP_AA",
    "left_pinky_MCP_FE",
    "left_thumb_MCP_FE",
    "left_index_PIP",
    "left_middle_PIP",
    "left_ring_PIP",
    "left_pinky_MCP_AA",
    "left_thumb_MCP_AA",
    "left_index_DIP",
    "left_middle_DIP",
    "left_ring_DIP",
    "left_pinky_PIP",
    "left_thumb_IP",
    "left_pinky_DIP",
]

NUM_ARM_DOFS: int = 7
NUM_HAND_DOFS: int = 22
NUM_DOFS: int = NUM_ARM_DOFS + NUM_HAND_DOFS

# ---------------------------------------------------------------------------
# Default joint positions (BFS order)
# Arm: [-1.571, 1.571, 0, 1.376, 0, 1.485, 1.308] (Sharpa mount offset)
# Hand: all zeros
# ---------------------------------------------------------------------------
DEFAULT_JOINT_POS_BFS: dict[str, float] = {
    "iiwa14_joint_1": -1.571,
    "iiwa14_joint_2": 1.571,
    "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376,
    "iiwa14_joint_5": 0.0,
    "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}

# ---------------------------------------------------------------------------
# Fingertip / palm body names
# ---------------------------------------------------------------------------
FINGERTIP_BODY_NAMES: list[str] = [
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
]
PALM_BODY_NAME: str = "iiwa14_link_7"

# Palm center offset from iiwa14_link_7 frame (meters)
PALM_CENTER_OFFSET: tuple[float, float, float] = (0.0, -0.02, 0.16)

# Fingertip offsets from distal phalanx frames (meters)
FINGERTIP_OFFSETS: list[tuple[float, float, float]] = [
    (0.02, 0.002, 0.0),  # thumb
    (0.02, 0.002, 0.0),  # index
    (0.02, 0.002, 0.0),  # middle
    (0.02, 0.002, 0.0),  # ring
    (0.02, 0.002, 0.0),  # pinky
]

# ---------------------------------------------------------------------------
# Keypoint offsets (object frame, unit cube corners, 4 points)
# Scaled by object_base_size * keypoint_scale / 2 at runtime
# ---------------------------------------------------------------------------
KEYPOINT_OFFSETS: list[tuple[float, float, float]] = [
    (+1.0, +1.0, +1.0),
    (+1.0, +1.0, -1.0),
    (-1.0, -1.0, +1.0),
    (-1.0, -1.0, -1.0),
]


@configclass
class SimToolRealEnvCfg(DirectRLEnvCfg):
    """Configuration for the SimToolReal Isaac Lab environment."""

    # --- Timing ---
    decimation: int = 1  # controlFrequencyInv=1, so every physics step
    episode_length_s: float = 10.0  # 600 steps at dt=1/60

    # --- Spaces ---
    action_space: int = NUM_DOFS  # 29 (arm 7 + hand 22)
    observation_space: int = 140  # sum of obs_list dims (to be verified)
    state_space: int = 0  # set >0 for asymmetric actor-critic

    # --- Simulation ---
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
        ),
    )

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=1.2,
        replicate_physics=True,
    )

    # --- Robot ---
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/robot/kuka_sharpa.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1000.0,
                angular_damping=0.0,  # IsaacGym default (Isaac Sim default is 0.05)
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.8, 0.0),
            joint_pos=DEFAULT_JOINT_POS_BFS,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["iiwa14_joint_.*"],
                stiffness={
                    "iiwa14_joint_1": 600.0,
                    "iiwa14_joint_2": 600.0,
                    "iiwa14_joint_3": 500.0,
                    "iiwa14_joint_4": 400.0,
                    "iiwa14_joint_5": 200.0,
                    "iiwa14_joint_6": 200.0,
                    "iiwa14_joint_7": 200.0,
                },
                damping={
                    "iiwa14_joint_1": 27.027,
                    "iiwa14_joint_2": 27.027,
                    "iiwa14_joint_3": 24.672,
                    "iiwa14_joint_4": 22.067,
                    "iiwa14_joint_5": 9.753,
                    "iiwa14_joint_6": 9.148,
                    "iiwa14_joint_7": 9.148,
                },
                effort_limit={
                    "iiwa14_joint_.*": 300.0,
                },
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["left_.*"],
                stiffness={
                    "left_1_thumb_CMC_FE": 6.95,
                    "left_thumb_CMC_AA": 13.2,
                    "left_thumb_MCP_FE": 4.76,
                    "left_thumb_MCP_AA": 6.62,
                    "left_thumb_IP": 0.9,
                    "left_2_index_MCP_FE": 4.76,
                    "left_index_MCP_AA": 6.62,
                    "left_index_PIP": 0.9,
                    "left_index_DIP": 0.9,
                    "left_3_middle_MCP_FE": 4.76,
                    "left_middle_MCP_AA": 6.62,
                    "left_middle_PIP": 0.9,
                    "left_middle_DIP": 0.9,
                    "left_4_ring_MCP_FE": 4.76,
                    "left_ring_MCP_AA": 6.62,
                    "left_ring_PIP": 0.9,
                    "left_ring_DIP": 0.9,
                    "left_5_pinky_CMC": 1.38,
                    "left_pinky_MCP_FE": 4.76,
                    "left_pinky_MCP_AA": 6.62,
                    "left_pinky_PIP": 0.9,
                    "left_pinky_DIP": 0.9,
                },
                damping={
                    "left_1_thumb_CMC_FE": 0.2868,
                    "left_thumb_CMC_AA": 0.4085,
                    "left_thumb_MCP_FE": 0.2039,
                    "left_thumb_MCP_AA": 0.2404,
                    "left_thumb_IP": 0.0419,
                    "left_2_index_MCP_FE": 0.2086,
                    "left_index_MCP_AA": 0.2460,
                    "left_index_PIP": 0.0424,
                    "left_index_DIP": 0.0350,
                    "left_3_middle_MCP_FE": 0.2086,
                    "left_middle_MCP_AA": 0.2460,
                    "left_middle_PIP": 0.0424,
                    "left_middle_DIP": 0.0350,
                    "left_4_ring_MCP_FE": 0.2086,
                    "left_ring_MCP_AA": 0.2460,
                    "left_ring_PIP": 0.0424,
                    "left_ring_DIP": 0.0350,
                    "left_5_pinky_CMC": 0.0278,
                    "left_pinky_MCP_FE": 0.2086,
                    "left_pinky_MCP_AA": 0.2460,
                    "left_pinky_PIP": 0.0424,
                    "left_pinky_DIP": 0.0350,
                },
            ),
        },
    )

    # --- Table ---
    # Table: spawned as static scene geometry, not tracked by RigidObject
    # (pattern from isaaclab_tasks/direct/automate/assembly_env.py)
    table_cfg: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/tables/table_narrow.usd",
    )
    table_pos: tuple[float, float, float] = (0.0, 0.0, 0.38)

    # --- Object ---
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/tools/hammer/claw_hammer.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                angular_damping=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.63),
        ),
    )
    # Object scale: (1,1,1) for fixed mesh objects; override for procedural tools
    object_scale_override: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # --- Control ---
    dof_speed_scale: float = 1.5
    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1

    # --- Reset ---
    reset_dof_pos_noise_arm: float = 0.1
    reset_dof_pos_noise_fingers: float = 0.1
    reset_dof_vel_noise: float = 0.5
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02

    # --- Reward ---
    lifting_rew_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_bonus_threshold: float = 0.15
    keypoint_rew_scale: float = 200.0
    distance_delta_rew_scale: float = 50.0
    reach_goal_bonus: float = 1000.0
    arm_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003
    fall_penalty: float = 0.0

    # --- Goal ---
    object_base_size: float = 0.04
    keypoint_scale: float = 1.5
    success_tolerance: float = 0.075
    success_steps: int = 10
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)
    target_volume_maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)
