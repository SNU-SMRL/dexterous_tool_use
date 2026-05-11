# Gate 4: SimToolReal → Isaac Lab Migration Plan

## 개요

SimToolReal의 `isaacgymenvs/tasks/simtoolreal/env.py` (5,314줄)을
Isaac Lab `DirectRLEnv` 기반으로 이식한다.

**원본:** `SimToolReal(VecTask)` — IsaacGym Preview 4, depth-first joints, xyzw quaternion
**대상:** `SimToolRealEnv(DirectRLEnv)` — Isaac Lab v2.3, breadth-first joints, wxyz quaternion

### Convention 원칙

환경 내부는 **전부 Isaac Lab native convention**으로 작성:
- Quaternion: **wxyz** (scalar first)
- Joint order: **BFS** (breadth-first)
- Convention 변환은 Gate 5에서 **policy wrapper**가 담당 (환경 코드에 혼재시키지 않음)

### 대상 파일 구조

```
scripts/simtoolreal_isaaclab/
├── __init__.py                  # Task registration
├── simtoolreal_env_cfg.py       # @configclass 환경 설정
├── simtoolreal_env.py           # DirectRLEnv 구현 (모든 로직 포함)
└── agents/
    ├── __init__.py
    └── rl_games_ppo_cfg.yaml    # rl_games 학습 설정
```

환경 코드가 1,000줄 이상으로 커지면 그때 rewards/observations를 분리.
Isaac Lab InHandManipulation 예제 (434줄)의 구조를 따른다.

**5,314줄을 1:1 이식하지 않는다.** Isaac Lab API가 대체하는 boilerplate를 제거하고,
핵심 로직(reward, observation, reset, action)만 이식한다.

---

## Phase 1: Scene Setup + Skeleton

**목표:** KUKA+Sharpa + 테이블 + 고정 오브젝트 1개를 로드하고, 모든 abstract method의 스텁을 작성하여 환경이 인스턴스화되는 것까지 확인

### 1-1. DirectRLEnv 필수 구현

DirectRLEnv는 6개 abstract method를 **모두** 구현해야 인스턴스화 가능:

| Method | Phase 1 (스텁) | 최종 구현 Phase |
|---|---|---|
| `_setup_scene()` | 로봇 + 테이블 + 오브젝트 로드 | Phase 1 |
| `_pre_physics_step(actions)` | `self.actions = actions.clone()` | Phase 2 |
| `_apply_action()` | pass | Phase 2 |
| `_get_observations()` | `{"policy": torch.zeros(N, obs_dim)}` | Phase 3 |
| `_get_rewards()` | `torch.zeros(N)` | Phase 4 |
| `_get_dones()` | `(torch.zeros(N, dtype=bool), timeout_check)` | Phase 4 |

### 1-2. 이식 대상 (원본 참조)

| 원본 | 내용 | Isaac Lab 대응 |
|---|---|---|
| `_create_envs()` L1768-2155 | Actor 생성: robot → object → table | `_setup_scene()`: `Articulation` + `RigidObject` |
| `populate_dof_properties()` utils.py | PD gains | `ArticulationCfg.actuators` → `ImplicitActuatorCfg` |
| Robot pose: `[0, 0.8, 0]` | 로봇 월드 위치 | `ArticulationCfg.init_state.pos` |
| Table pose: `[0, 0.0, tableResetZ]` | 테이블 위치 | `RigidObjectCfg.init_state.pos` |

### 1-3. SimulationCfg: PhysX 기본값을 Phase 1에서 바로 설정

IsaacGym과 Isaac Sim의 기본값 차이를 **처음부터** 맞춤.
Phase 5까지 미루면 Phase 2-4의 모든 물리 테스트가 다른 조건에서 실행됨.

```python
sim: SimulationCfg = SimulationCfg(
    dt=1/60,
    render_interval=1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=0.5,
        dynamic_friction=0.5,
    ),
    physx=sim_utils.PhysxCfg(
        solver_type=1,                          # TGS (원본과 동일)
        max_position_iteration_count=8,         # 원본 config
        max_velocity_iteration_count=0,         # 원본 config
        bounce_threshold_velocity=0.2,          # 원본 config
        max_depenetration_velocity=1000.0,      # 원본 config
    ),
)
```

USD 에셋의 rigid body properties에서도 override:
- Angular Damping: 0.0 (Isaac Sim 기본 0.05 → IsaacGym 기본 0.0)
- Contact Offset: 0.002 (원본 config)

### 1-4. 에셋 경로 (Phase 1은 고정 오브젝트)

- Robot: `assets/usd/robot/kuka_sharpa.usd`
- Table: `assets/usd/tables/table_narrow.usd`
- Object: `assets/usd/tools/` 중 1개 (예: hammer) — Phase 2.5에서 multi-asset으로 전환

### 1-5. DOF 설정

원본 `populate_dof_properties()`의 PD gains를 `ImplicitActuatorCfg`로:
```
Arm (0-6):  stiffness=[600,600,500,400,200,200,200], damping=[27,27,24.7,22.1,9.7,9.1,9.1]
Hand (7-28): stiffness=[0.9-13.2], damping=[0.028-0.408]
```

**Blue robot (시각화용):** 이식하지 않음.

### 1-6. 검증

```bash
source .venv/bin/activate
python scripts/simtoolreal_isaaclab/simtoolreal_env.py --num_envs 4 --livestream 1
```

- [ ] 환경이 에러 없이 인스턴스화됨 (모든 스텁 동작)
- [ ] 로봇이 렌더링됨 (29 joints)
- [ ] 테이블이 올바른 위치
- [ ] 오브젝트가 테이블 위에 있음
- [ ] `robot.data.joint_names`가 `JOINT_NAMES_ISAACLAB`과 일치 (**Gate 2 TODO 해소**)
- [ ] VRAM 사용량 기록

### 1-7. 리스크

- USD collision mesh가 다르게 동작 → Isaac Lab viewer에서 collision 시각화로 확인
- ImplicitActuator가 원본 PD gains와 정확히 일치하지 않을 수 있음 → 스텝 응답 비교
- 테이블/오브젝트 물리 속성(mass, friction) 불일치 → USD 내 값 확인 후 config override

---

## Phase 2: Reset + Random Action Step

**목표:** 환경 reset과 랜덤 액션으로 step이 정상 동작

### 2-1. 이식 대상

| 원본 | Isaac Lab 대응 |
|---|---|
| `reset_idx()` L3508-3690 | `_reset_idx(env_ids)` → `write_joint_state_to_sim()`, `write_root_pose_to_sim()` |
| `pre_physics_step()` L3720-3891 | `_pre_physics_step()` + `_apply_action()` |

### 2-2. 조인트 순서

Arm (0-6)은 DFS/BFS 동일. Hand (7-28)만 순서 다름.
`scripts/joint_remapping.py`에 매핑 존재.

**변환이 필요한 곳:**
- Default DOF positions 배열 순서
- PD gains 배열 순서 (hand 내부만)

**변환 불필요:**
- Action split `[:7]` / `[7:29]` — arm이 여전히 앞 7개
- DOF limits — `robot.data` API가 BFS 순서로 제공

### 2-3. 액션 처리 로직

```python
def _pre_physics_step(self, actions: torch.Tensor) -> None:
    self.actions = actions.clone()

def _apply_action(self) -> None:
    # Arm: relative incremental position control
    arm_targets = self.robot.data.joint_pos[:, :7] + self.speed_scale * self.dt * self.actions[:, :7]
    arm_targets = self.arm_ma * arm_targets + (1 - self.arm_ma) * self.prev_targets[:, :7]

    # Hand: [-1, 1] → joint limits, with smoothing
    hand_targets = scale_transform(self.actions[:, 7:29], self.hand_lower, self.hand_upper)
    hand_targets = self.hand_ma * hand_targets + (1 - self.hand_ma) * self.prev_targets[:, 7:29]

    targets = torch.cat([arm_targets, hand_targets], dim=-1)
    targets = saturate(targets, self.dof_lower, self.dof_upper)

    self.prev_targets[:] = self.cur_targets
    self.cur_targets[:] = targets
    self.robot.set_joint_position_target(targets)
```

### 2-4. Reset 로직

```python
def _reset_idx(self, env_ids: torch.Tensor) -> None:
    super()._reset_idx(env_ids)

    # Joint positions: default + noise (arm/hand 별도 noise coefficient)
    noise = torch.rand_like(default_pos[env_ids]) * noise_coeff
    joint_pos = default_pos[env_ids] + noise * (upper - default_pos[env_ids])
    joint_vel = torch.zeros_like(joint_pos)
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Object: table 위 랜덤 위치
    object_pose = self.object_init_pose.clone()
    object_pose[:, 0] += noise_x  # X noise
    object_pose[:, 1] += noise_y  # Y noise
    self.object.write_root_pose_to_sim(object_pose, env_ids=env_ids)
    self.object.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6), env_ids=env_ids)

    # Goal sampling
    self._resample_goal(env_ids)

    # Internal buffers reset
    self.prev_targets[env_ids] = joint_pos
    self.cur_targets[env_ids] = joint_pos
```

### 2-5. 검증

```bash
# --livestream으로 시각 확인 + tensorboard에 step 수 로깅
python scripts/simtoolreal_isaaclab/simtoolreal_env.py --num_envs 4 --livestream 1
```

- [ ] `env.reset()` 에러 없음
- [ ] 랜덤 액션 100 step 정상 진행
- [ ] 로봇 관절이 움직이는 것을 livestream으로 확인
- [ ] 오브젝트가 중력/접촉에 반응
- [ ] Reset 후 로봇이 default pose 근처로 복귀

---

## Phase 2.5: Multi-Asset Object Pipeline

**목표:** 고정 오브젝트 1개 → 다양한 tool object로 전환.
Phase 4 (Reward)의 `keypoint_offset`이 `object_scale`에 의존하므로, reward 이식 전에 해결.

### 2.5-1. 원본 동작

SimToolReal의 procedural generation:
1. `generate_objects.py`가 6 tool type × 100 variant = **600개 URDF** 생성 (임시 디렉토리)
2. 각 URDF에 크기/밀도가 baked-in (box + cylinder 조합)
3. `gym.load_asset()`로 600개 로드
4. Env마다 `env_idx % 600`으로 다른 에셋 할당

### 2.5-2. Isaac Lab 대응: 두 가지 경로

**경로 A: `MultiAssetSpawnerCfg` + Isaac Lab primitive shapes**

handle-head 도구는 box + cylinder 조합. Isaac Lab의 `CuboidCfg`/`CylinderCfg`로 직접 생성 가능.
URDF→USD 변환 파이프라인 불필요.

```python
spawn=sim_utils.MultiAssetSpawnerCfg(
    assets_cfg=[
        sim_utils.CuboidCfg(size=(...)),   # variant 1
        sim_utils.CylinderCfg(radius=...), # variant 2
        ...
    ],
    random_choice=True,
)
```

요구사항:
- `InteractiveSceneCfg(replicate_physics=False)` — scene 생성 느려짐
- 600개 variant를 primitive cfg 리스트로 생성하는 코드 필요
- collision/mass properties를 primitive별로 설정

**경로 B: Pre-generate USDs + `MultiUsdFileCfg`**

`generate_objects.py`로 URDF 생성 → batch URDF→USD 변환 → USD 파일을 `MultiUsdFileCfg`로 로드.

```python
spawn=sim_utils.MultiUsdFileCfg(
    usd_path=["tools/hammer_001.usd", "tools/hammer_002.usd", ...],
    random_choice=True,
)
```

요구사항:
- 변환 스크립트 작성 (기존 `convert_urdf_to_usd.sh` 확장)
- 600개 USD 파일 생성 (디스크 공간)
- `replicate_physics=False`

**경로 결정:** Phase 2.5 시작 시 경로 A를 먼저 시도 (URDF 의존성 제거).
handle-head가 단순 primitive로 표현 불가능하면 경로 B로 전환.

### 2.5-3. `object_scale` 버퍼

어떤 경로든, env마다 `object_scale` (3D) 버퍼를 유지해야 함:
- Keypoint offset 계산에 사용 (Phase 4)
- Observation에 포함 (`object_scales`)
- `_setup_scene()`에서 variant별 scale을 기록

### 2.5-4. `dextoolbench` 의존성 해소

`from dextoolbench.objects import NAME_TO_OBJECT` — simtoolreal 내부 모듈.
필요한 상수 (6개 tool type의 이름, 기본 크기)만 `simtoolreal_env_cfg.py`에 복사.

### 2.5-5. 검증

- [ ] 4개 이상의 env에서 서로 다른 오브젝트가 로드됨 (시각적 확인)
- [ ] `object_scale` 버퍼가 env마다 올바른 값
- [ ] `replicate_physics=False` 시 scene 생성 시간 기록 (8192 env 기준)
- [ ] 오브젝트 물리 동작이 Phase 2와 동일 (중력, 접촉)

---

## Phase 3: Observation

**목표:** 원본과 동일한 구조의 observation 벡터 구성 (Isaac Lab native convention)

### 3-1. Convention

**환경은 전부 Isaac Lab native로 출력:**
- Quaternion: wxyz
- Joint order: BFS

Pretrained policy 호환을 위한 변환 (wxyz→xyzw, BFS→DFS)은 **Gate 5의 policy wrapper**에서 처리.
환경 코드에 convention 변환을 넣지 않는다.

### 3-2. Observation 항목

**Policy obs** (원본 `obsList` 기준):

| Obs 이름 | 차원 | Isaac Lab 소스 | 비고 |
|---|---|---|---|
| `joint_pos` | 29 | `robot.data.joint_pos` → `unscale_transform` | BFS 순서 |
| `joint_vel` | 29 | `robot.data.joint_vel` | BFS 순서 |
| `prev_action_targets` | 29 | 자체 버퍼 `self.prev_targets` | BFS 순서 |
| `palm_pos` | 3 | `robot.data.body_pos_w[:, palm_idx]` | 오프셋 `[0, -0.02, 0.16]` 적용 |
| `palm_rot` | 4 | `robot.data.body_quat_w[:, palm_idx]` | **wxyz** |
| `object_rot` | 4 | `object.data.root_quat_w` | **wxyz** |
| `fingertip_pos_rel_palm` | 15 | `robot.data.body_pos_w[:, ft_ids] - palm_pos` | 5 fingertip × 3, 오프셋 적용 |
| `keypoints_rel_palm` | 12 | `quat_apply(obj_quat, kp_offsets) + obj_pos - palm_pos` | `math.quat_apply()` wxyz |
| `keypoints_rel_goal` | 12 | `goal_keypoints - current_keypoints` | — |
| `object_scales` | 3 | Phase 2.5에서 설정한 버퍼 | — |

**Critic state** (asymmetric, `stateList` 기준): policy obs + 추가 항목

| 추가 항목 | 차원 | 비고 |
|---|---|---|
| `palm_vel` | 6 | lin_vel + ang_vel |
| `object_vel` | 6 | lin_vel + ang_vel |
| `closest_keypoint_max_dist` | 1 | 에피소드 내 최소값 추적 |
| `closest_fingertip_dist` | 5 | 에피소드 내 최소값 추적 |
| `lifted_object` | 1 | bool → float |
| `progress` | 1 | `log(step / 10 + 1)` |
| `successes` | 1 | `log(successes + 1)` |
| `reward` | 1 | `0.01 * rew_buf` |

### 3-3. Body Name → Index 매핑

Phase 1에서 `robot.find_bodies()`로 확인:
```python
palm_idx = robot.find_bodies("iiwa14_link_7")[0]
fingertip_ids = robot.find_bodies([
    "left_thumb_DP", "left_index_DP", "left_middle_DP",
    "left_ring_DP", "left_pinky_DP"
])[0]
```

### 3-4. 검증

```bash
# obs 값을 로깅하고 플롯으로 비교
python scripts/simtoolreal_isaaclab/simtoolreal_env.py --num_envs 4 --livestream 1
```

- [ ] `_get_observations()["policy"]` shape = `(num_envs, sum(obs_dims))`
- [ ] `_get_states()` shape = `(num_envs, sum(state_dims))`
- [ ] 100 step 동안 obs 값에 NaN/inf 없음
- [ ] `palm_pos` 값이 로봇 위치 근처 (world frame 기준 합리적 범위)
- [ ] `keypoints_rel_palm` 값이 ~0.1m 이내 (테이블 위 오브젝트 ↔ 팔 끝)

---

## Phase 4: Reward + Done + Goal

**목표:** 원본과 동일한 reward 함수, termination 조건, goal sampling

### 4-1. Reward 구성

```
total_reward = fingertip_delta_rew * 50.0      # 핑거팁 접근 (리프팅 전만)
             + lifting_rew * 20.0              # Z축 리프팅 (clamp 0~0.5m)
             + lifting_bonus (300.0, 1회)       # 0.15m 이상 리프팅 시
             + keypoint_rew * 200.0            # 키포인트 정렬 (리프팅 후만)
             - arm_action_penalty * 0.03       # |arm_dof_vel| 합
             - hand_action_penalty * 0.003     # |hand_dof_vel| 합
             + reach_goal_bonus (1000.0)       # 목표 도달 시
```

각 보상 항목은 `_compute_*()` private method로 분리하되 같은 파일에 유지.

### 4-2. Termination + Goal 성공 = Full Reset

**단순화:** goal 성공 시 full reset. Isaac Lab 프레임워크 그대로 사용.

원본은 goal 달성 시 환경을 유지하고 goal만 재샘플링 (multi-goal per episode)하지만,
이는 학습 효율을 위한 최적화. 먼저 full reset으로 동작을 확보한 뒤,
학습 성능이 부족하면 `_reset_idx()` 내에 goal-only 분기를 추가.

```python
def _get_rewards(self) -> torch.Tensor:
    ...
    # Goal 달성 감지 (reward 계산용)
    near_goal = keypoints_max_dist <= self.success_tolerance
    self.near_goal_steps[near_goal] += 1
    self.near_goal_steps[~near_goal] = 0
    self.goal_reached = self.near_goal_steps >= self.success_steps
    ...
    return total_reward

def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    fallen = self.object.data.root_pos_w[:, 2] < 0.1
    too_far = torch.norm(palm_pos - object_pos, dim=-1) > 1.5
    terminated = fallen | too_far | self.goal_reached  # goal 성공도 terminated

    time_out = self.episode_length_buf >= self.max_episode_length
    return terminated, time_out
```

`_reset_idx()`는 Phase 2에서 구현한 full reset 그대로 사용.

**추후 (Gate 5 결과에 따라):** 학습이 너무 느리면 goal-only reset 도입.
`_reset_idx()` 안에서 `goal_reached` env만 goal 재샘플링 + 버퍼 리셋,
나머지는 full reset. 코드 변경량은 적음.

### 4-3. Goal Sampling (`_resample_goal`)

원본 `goalSamplingType: "delta"`:
- 현재 오브젝트 위치 기준 0.1m 반경, 90도 회전 범위 내 샘플링
- `success_tolerance`: 0.075 → curriculum으로 0.01까지 감소 (Phase 5b)

```python
def _resample_goal(self, env_ids: torch.Tensor) -> None:
    # 현재 오브젝트 위치에서 delta 샘플링
    obj_pos = self.object.data.root_pos_w[env_ids]
    obj_quat = self.object.data.root_quat_w[env_ids]  # wxyz

    delta_pos = sample_uniform(-0.1, 0.1, (len(env_ids), 3), device=self.device)
    goal_pos = obj_pos + delta_pos
    goal_pos = torch.clamp(goal_pos, self.target_volume_min, self.target_volume_max)

    delta_quat = random_orientation(len(env_ids), device=self.device)  # wxyz
    goal_quat = quat_mul(delta_quat, obj_quat)

    self.goal_pos[env_ids] = goal_pos
    self.goal_quat[env_ids] = goal_quat
    self.goal_keypoints[env_ids] = self._compute_keypoints(goal_pos, goal_quat, env_ids)
```

### 4-4. 검증

```bash
python scripts/simtoolreal_isaaclab/simtoolreal_env.py --num_envs 16 --livestream 1
```

- [ ] Reward 값이 합리적 범위 (-10 ~ 1500), NaN/inf 없음
- [ ] 오브젝트 낙하 시 `terminated=True`
- [ ] Goal 달성 시 `terminated=True` → full reset 정상 동작
- [ ] 에피소드 타임아웃 = `episodeLength` (600) step
- [ ] Reward 항목별 플롯: lifting, keypoint, penalty 각각 로깅

---

## Phase 5: Domain Randomization + Sim2Real 세부 로직

**목표:** 원본과의 동작 일치도를 높이는 세부 기능

### 5a: 필수 항목 (학습 성능에 직접 영향)

- [ ] Random forces/torques on object (리프팅 후에만, `forceScale: 20.0`)
- [ ] Action delay (큐 기반, 최대 3 step)
- [ ] Observation delay (최대 3 step)
- [ ] Object state noise (xyz std 0.01, rotation 5도)
- [ ] Joint velocity observation noise (std 0.1)

### 5b: 선택 항목 (Gate 5 결과에 따라)

- [ ] DOF property randomization (stiffness, damping ×0.7-1.3 log-uniform)
- [ ] Mass randomization (robot, object ×0.7-1.3)
- [ ] Friction randomization (×0.7-1.3)
- [ ] Tolerance curriculum (0.075 → 0.01, ×0.9 per 3000 episodes)
- [ ] Observation dropout (tyler curriculum)

### 5-2. 검증

- [ ] Random force 시 오브젝트 합리적으로 반응
- [ ] Action delay 적용 전후 reward 변화 확인 (플롯)
- [ ] DR on/off 시 학습 곡선 비교 (Gate 5 이후)

---

## 실행 순서

```
Phase 1 (반나절): Scene + Skeleton
├── @configclass 작성 (SimulationCfg + PhysX 기본값 포함)
├── _setup_scene(): robot + table + 고정 object
├── 6개 abstract method 스텁 구현
├── joint_names 실증 대조 (Gate 2 TODO)
└── 검증: --livestream 렌더링 + VRAM

Phase 2 (반나절): Reset + Action
├── _reset_idx(): joint/object reset + goal sampling
├── _apply_action(): arm relative + hand scale control
└── 검증: 랜덤 액션 100 step + livestream

Phase 2.5 (반나절~1일): Multi-Asset Objects
├── 경로 결정: MultiAssetSpawner primitives vs pre-gen USD
├── object_scale 버퍼 구축
├── dextoolbench 상수 복사
└── 검증: 다수 env에서 서로 다른 오브젝트 확인

Phase 3 (반나절): Observation
├── _get_observations(): policy obs (wxyz, BFS)
├── _get_states(): critic state (asymmetric)
├── body name → index 매핑 검증
└── 검증: obs shape + 값 범위 플롯

Phase 4 (반나절~1일): Reward + Done + Goal
├── _get_rewards(): 5개 reward 항목
├── _get_dones(): 낙하/거리/시간 초과
├── Goal-only reset (_get_rewards 내부)
├── Goal sampling (delta 방식)
└── 검증: reward 항목별 플롯 + goal reset 동작

Phase 5a (반나절): 필수 DR
├── Random forces, action/obs delay, state noise
└── 검증: delay 전후 비교

Phase 5b (Gate 5 이후): 선택 DR
├── Property randomization, curriculum
└── 검증: 학습 곡선 비교
```

예상 소요: Phase 1-4 = 2~3일, Phase 5a = 반나절, 총 ~3일

---

## 의존 파일 매핑

| 원본 파일 | 이식 방법 |
|---|---|
| `env.py` (5,314줄) | → `simtoolreal_env.py` (Isaac Lab API로 대폭 축소) |
| `utils.py` (309줄) | → 인라인 (populate_dof_properties → ImplicitActuatorCfg, tolerance_curriculum → Phase 5b) |
| `generate_objects.py` (833줄) | → Phase 2.5에서 결정 (경로 A: 불필요, 경로 B: 복사+확장) |
| `object_size_distributions.py` (290줄) | → Phase 2.5에서 복사 (크기/밀도 분포 필요) |
| `adjacent_links.py` (88줄) | → 복사 (collision filter용) |
| `observation_action_utils_sharpa.py` (734줄) | → 불필요 (`isaaclab.utils.math` + `robot.data` API 대체) |
| `torch_jit_utils.py` | → `isaaclab.utils.math`로 대체 (`quat_apply`, `scale_transform`, etc.) |
| `SimToolReal.yaml` (330줄) | → `simtoolreal_env_cfg.py` (`@configclass`) |
| `dextoolbench.objects` | → 필요한 상수만 `simtoolreal_env_cfg.py`에 복사 |

---

## 리스크 및 완화

| 리스크 | 영향 | 완화 |
|---|---|---|
| Multi-asset + `replicate_physics=False` 성능 | Phase 2.5 | scene 생성 시간 측정, 필요 시 variant 수 축소 (600→60) |
| handle-head를 primitive로 표현 불가 | Phase 2.5 | 경로 B (pre-gen USD)로 전환 |
| PhysX 기본값 차이 | Phase 1-4 전체 | Phase 1 SimulationCfg에서 즉시 override |
| Goal-only reset 필요 시 학습 효율 | Gate 5 | 먼저 full reset, 성능 부족 시 `_reset_idx()` 분기 추가 |
| Pretrained policy convention 불일치 | Gate 5 | 환경은 native convention, policy wrapper에서 변환 |
| `dextoolbench` import 불가 | Phase 1 | 상수만 복사, 외부 의존성 제거 |
| 팜/핑거팁 오프셋 누락 | Phase 3 | 원본 하드코딩 값 그대로 적용, body state와 비교 검증 |
