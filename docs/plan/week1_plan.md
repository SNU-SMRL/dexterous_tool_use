# Month 1: Isaac Lab 환경 + Impedance Controller

## 프로젝트 방향 (2026-05-11 전환)

**이전:** SimToolReal rollout → GR00T N1.7 post-training → language-conditioned tool use
**현재:** UniDex-VLA (frozen) + Adaptive Impedance Control for contact-rich dexterous tool use

**핵심 아이디어:** UniDex-VLA는 WHERE (position target)를 알려주고, Impedance Adapter MLP는 HOW (adaptive K, D gain)를 결정한다.

---

## Gate 0: UniDex-VLA Inference + Feature 추출

**목표:** UniDex-VLA pretrained model에서 action chunk와 internal feature를 추출할 수 있는지 확인

### 0-1. 환경 셋업

```bash
source .venv/bin/activate

# UniDex 클론
git clone --recurse-submodules https://github.com/unidex-ai/UniDex.git
cd UniDex
uv pip install -e .

# Pretrained weights
# HuggingFace: UniDex-ai/UniDex
```

### 0-2. Inference 테스트

UniDex에 standalone inference 스크립트는 없음. `PointCloudUniDexInference` 클래스 직접 사용:

```python
from src.unidex.unidex import PointCloudUniDexInference

model = PointCloudUniDexInference(config)
# checkpoint 로드: torch.load() → strip "policy." prefix → load_state_dict()

# Inputs:
#   pointcloud: (B, pcd_horizon, pointcloud_size, 6)  -- xyz + rgb
#   state:      (B, cond_steps, 82)                    -- FAAS proprioception
#   prompt:     list[str]                               -- language instruction

action = model.infer_action(pointcloud, state, prompt)
# Output: (B, 30, 82) -- 30-step action chunk in FAAS format
```

### 0-3. Internal Feature 추출

**z (latent representation):**
- `infer_action()` 내부에서 `self.joint_model()` 호출 후 KV caches에 VLM+proprio 표현이 담김
- 방법 1: `infer_action()`을 수정하여 KV caches를 반환
- 방법 2: `self.joint_model`에 PyTorch forward hook 등록

**a_future (action chunk):**
- `infer_action()`의 직접 반환값 (B, 30, 82)
- Flow matching 10-step Euler denoising으로 생성

### 0-4. FAAS Action Space 확인

FAAS 82-dim 구성:
- Wrist poses: 9D × 2 hands = 18
- Right hand joints: 32 (MAPPED_JOINT_DIM, 실제 사용 27)
- Left hand joints: 32

LEAP Hand (16-DoF)의 FAAS 매핑:
- `src/assets/utils/hand_utils.json`에 JOINT_MAP 정의
- 16개 native joint → 32-dim canonical vector (미사용 슬롯은 0)

### 통과 기준
- [ ] Pretrained model 로드 + inference 실행 성공
- [ ] FAAS action output (B, 30, 82) 확인
- [ ] KV caches (z) 추출 가능 확인
- [ ] LEAP Hand action이 FAAS 82-dim 중 어느 슬롯에 매핑되는지 확인

### 실패 시
- Dependency 충돌 → 별도 venv 구성 (UniDex는 Isaac Lab과 독립)
- z 추출 불가 → action chunk만 사용 (a_future만으로 adapter 학습)
- Pretrained weights 접근 불가 → HuggingFace gated repo 승인 확인

---

## Gate 1: LEAP Hand + Franka Isaac Lab 환경

**목표:** LEAP Hand 16-DoF + Franka arm을 Isaac Lab DirectRLEnv로 구축

### 1-1. LEAP Hand URDF 확보

```bash
# LEAP Hand Sim repo (URDF 공개)
# UniDex repo에도 포함: HandAdapter/urdf/base/Leap/{left,right}/main.urdf
# 16-DoF, 4 fingertip links: realtip, realtip_2, realtip_3, realtip_4
```

### 1-2. URDF → USD 변환

```bash
# Isaac Lab CLI 사용 (Gate 2 경험 활용)
# Robot: --fix-base (kinematic chain 보존)
# 주의: instanceable prims에 collision_props 적용 불가 이슈 (Gate 4에서 해결 경험 있음)
```

### 1-3. DirectRLEnv 구축

Isaac Lab 환경 구조:
```
scripts/impedance_env/
├── __init__.py
├── env_cfg.py      # scene, actuator, sensor config
└── env.py          # DirectRLEnv 구현
```

참고: 기존 Gate 4에서 DirectRLEnv 구축 경험 있음 (SimToolReal env).
공식 예제 패턴: `isaaclab_tasks/direct/inhand/inhand_env.py`

### 1-4. ContactSensor 설정

```python
# F_contact 읽기용 ContactSensor 설정
# Fingertip body에 ContactSensor 부착
# 매 step에서 net_forces_w 또는 force_matrix_w 읽기
```

### 통과 기준
- [ ] LEAP Hand + Franka arm articulation이 Isaac Lab에서 로드됨
- [ ] 16개 hand joint + 7개 arm joint = 23 joint 확인
- [ ] Reset 정상 동작
- [ ] Random action으로 step 정상 진행
- [ ] ContactSensor에서 F_ext 읽기 성공
- [ ] VRAM 사용량 기록

### 실패 시
- LEAP Hand URDF 변환 문제 → UniDex repo의 URDF 직접 사용
- SM_120 NVRTC 에러 → Isaac Sim 5.1 venv 사용 (Gate 5 경험)
- LEAP Hand 대체 → Shadow Hand (Isaac Lab 기본 제공, UniDex도 지원)

---

## Gate 2: Admittance Controller 구현

**목표:** PD actuator 위에 admittance layer를 구현하여 variable impedance control 확인

### 2-1. Admittance Control 구조

Position-controlled actuator 기준의 admittance control:

```
F_ext (ContactSensor)
    ↓
M_d · ë + D_d · ė + K_d · e = F_ext    (가상 dynamics)
    ↓
Δx (위치 보정)
    ↓
x_cmd = x_target + Δx → PD controller
```

- **M_d**: 가상 관성 (고정, 튜닝 파라미터)
- **D_d**: 댐핑 (adapter가 생성)
- **K_d**: 강성 (adapter가 생성)
- **e**: x_target - x_current (위치 에러)

### 2-2. Arm vs Finger 분리

**Arm (cartesian impedance):**
- Franka 7-DoF → cartesian 6-DoF impedance
- Jacobian transpose로 joint torque 변환
- 또는 operational space control

**Finger (per-joint impedance):**
- LEAP Hand 16-DoF → joint-level impedance
- 그룹화: 엄지 4-DoF (독립) + 나머지 4손가락 (3-DoF × 4, 그룹 K/D)
- τ_finger = K_f · (q_target - q) + D_f · (q̇_target - q̇)

### 2-3. 검증

```python
# Scripted trajectory로 검증
# 1. 고정 K/D로 안정적 tracking 확인
# 2. K를 높이면 뻣뻣, 낮추면 유연한지 확인
# 3. D를 높이면 진동 감소하는지 확인
# 4. 외력 인가 시 admittance 응답 확인
```

### 통과 기준
- [ ] Variable K/D로 안정적 trajectory tracking
- [ ] K 변화에 따른 stiffness 변화 확인 (정성적)
- [ ] D 변화에 따른 damping 변화 확인 (정성적)
- [ ] Arm cartesian impedance + finger joint impedance 동시 동작
- [ ] F_ext에 대한 compliant 응답 확인

### 실패 시
- Isaac Lab actuator 모델과 충돌 → effort-controlled joint로 전환 (torque 직접 제어)
- Cartesian impedance 불안정 → joint-level impedance only로 단순화
- 전공 역량으로 해결 가능 (GNC 배경)

---

## Gate 3: Contact-Rich 태스크 환경

**목표:** 2-3개 contact-rich 태스크 환경을 Isaac Lab에서 구축

### 3-1. 태스크 우선순위

| 순서 | 태스크 | 이유 |
|------|--------|------|
| 1 | **젓기 (Stirring)** | 지속적 저항, phase transition 완만, 물리 안정성 높음. 디버깅에 적합 |
| 2 | **망치질 (Hammering)** | 급격한 phase transition (스윙 → 충돌). Anticipatory grip 테스트 |
| 3 | 병뚜껑 돌리기 | 지속 토크 + 갑작스러운 저항 소실. 선택 사항 |

### 3-2. 젓기 환경

에셋:
- 냄비: cylinder 또는 simple mesh (RigidObject, kinematic)
- 스푼: simple mesh (RigidObject, hand에 grasp)
- 유체: 없음 (저항을 force로 모사, 또는 granular particle)

보상:
- 스푼 tip의 원형 궤적 추종
- 냄비 벽면 충돌 패널티
- Grasp 유지 보상

### 3-3. 망치질 환경

에셋:
- 망치: simple mesh (handle + head, RigidObject)
- 못: cylinder (target, kinematic)
- 보드: box (static)

보상:
- 못에 대한 수직 충격력
- Grasp 유지 (충돌 후 슬립 패널티)
- 정확도 (못 중심 타격)

### 3-4. PhysX 파라미터 튜닝

충격 태스크에서 주의:
- `solver_position_iteration_count`: 기본 4 → 8~16으로 증가
- `solver_velocity_iteration_count`: 기본 1 → 4로 증가
- `max_depenetration_velocity`: 과도한 반발 방지
- `contact_offset`, `rest_offset`: 접촉 감지 민감도

### 통과 기준
- [ ] 젓기 환경에서 scripted trajectory로 태스크 완수 가능
- [ ] 망치질 환경에서 충돌 시 물리 안정적 (폭발/관통 없음)
- [ ] ContactSensor에서 충격력 로깅 가능
- [ ] 환경 reset 정상 동작

### 실패 시
- 망치질 물리 불안정 → 젓기 환경에만 집중 (1개 태스크로도 충분)
- 복잡한 tool mesh → simple primitive (cylinder + box)로 대체
- Contact solver 발산 → solver iteration 증가 + time step 감소

---

## 실행 순서

```
Day 1-2: UniDex-VLA 확인 (Gate 0)
├── 클론 + 의존성 설치
├── Pretrained model 로드 + inference 실행
├── FAAS action output 확인
└── Internal feature (z, a_future) 추출 테스트

Day 3-5: LEAP Hand Isaac Lab 환경 (Gate 1)
├── URDF → USD 변환
├── DirectRLEnv skeleton (scene/reset/obs)
├── ContactSensor 설정
└── Random action step 테스트

Day 6-8: Admittance controller (Gate 2)
├── Joint-level impedance 구현 (finger)
├── Cartesian impedance 구현 (arm)
├── Variable K/D 검증
└── F_ext compliant 응답 확인

Day 9-12: Contact-rich 태스크 (Gate 3)
├── 젓기 환경 구축 + 검증
├── 망치질 환경 시도
├── PhysX 파라미터 튜닝
└── Scripted trajectory baseline 측정
```

---

## Month 1 종료 시 판단 기준

| 결과 | 다음 행동 |
|------|-----------|
| Gate 0-3 모두 통과 | Month 2: UniDex-VLA → Isaac Lab 연동 + gain optimization |
| Gate 0 통과, Gate 1 실패 | Shadow Hand로 대체 (Isaac Lab 기본 제공, UniDex 지원) |
| Gate 0 실패 (feature 추출 불가) | Action chunk만 사용 (z 없이 a_future + F_contact → K/D) |
| Gate 2 실패 (admittance 불안정) | Joint-level PD impedance로 단순화 |
| Gate 3 실패 (물리 불안정) | 젓기 환경 1개에 집중 |

---

## 기존 작업에서 활용하는 것

| 기존 경험 | 활용 방식 |
|-----------|-----------|
| Isaac Lab 설치 + DirectRLEnv 구축 (Gate 4) | 동일 패턴으로 LEAP Hand env 구축 |
| URDF → USD 변환 (Gate 2) | 동일 스크립트/경험으로 LEAP Hand 변환 |
| SM_120 NVRTC 이슈 대응 (Gate 5) | Isaac Sim 5.1 venv 전략 그대로 적용 |
| RigidObject vs static spawn 판단 (Gate 4) | 냄비/보드는 static spawn, tool은 RigidObject |
| API 검증 습관 (CLAUDE.md) | unfamiliar API는 소스에서 확인 후 사용 |

---

## 리스크 로그

| 리스크 | 영향 | 완화 |
|--------|------|------|
| UniDex-VLA 의존성이 Isaac Lab과 충돌 | Gate 0 | 별도 venv로 분리 (inference만 사용) |
| LEAP Hand URDF에 collision mesh 누락 | Gate 1 | UniDex repo URDF 사용, 또는 Shadow Hand 대체 |
| Admittance controller + PD actuator 상호작용 불안정 | Gate 2 | Effort-controlled joint로 전환 |
| 망치질 contact solver 발산 | Gate 3 | 젓기 환경에 집중, solver iteration 증가 |
| 전체 Month 1에 Gate 0-1만 통과 | 일정 지연 | Gate 2-3는 Month 2 초반으로 이월 가능 |
