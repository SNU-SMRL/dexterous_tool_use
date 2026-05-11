# Week 1: SimToolReal → Isaac Lab Migration + GR00T 검증

## 주의사항

**IsaacGym Preview 4는 RTX 5080(SM_120)에서 동작 불가.**
- `libPhysXGpu_64.so`에 SM_80까지만 커널 포함, deprecated, 업데이트 없음
- Isaac Lab으로 이식하여 Python 3.10 단일 환경으로 통일

---

## Gate 1: Isaac Lab 설치 + 기본 동작 확인

**목표:** Isaac Lab 설치, RTX 5080에서 dexterous hand 예제 환경 실행 확인

### 1-1. 환경 셋업

```bash
# uv 설치 (없으면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 메인 venv: Python 3.10 (Isaac Sim 4.5 + GR00T 통합)
uv venv --python 3.10 .venv
source .venv/bin/activate

# Isaac Lab 설치
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html
# Isaac Sim 4.5 + Isaac Lab (main branch)
```

**Multi-venv (Gate 5 blocker 해결용, 2026-05-11 추가):**
- `.venv-5.1` (Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3.x) — PhysX SM_120 지원
- `.venv-6.0` (Python 3.12, Isaac Sim 6.0 beta, Isaac Lab 3.0-beta) — 렌더링 SM_120 지원

### 1-2. Dexterous Hand 예제 실행

```bash
# Shadow Hand 환경 테스트
python -m isaaclab.scripts.run_env --task Isaac-Shadow-Hand-Over-Direct-v0 --num_envs 16
```

### 통과 기준
- [x] Isaac Lab이 RTX 5080에서 정상 실행됨 (Isaac Sim 4.5 + Isaac Lab main branch)
- [ ] Shadow Hand가 렌더링되고 관절이 움직임 (livestream 미동작 — Blocker 2)
- [ ] VRAM 사용량 기록

### 실패 시
- Isaac Sim/Lab 설치 실패 → CUDA 12.x 호환 확인, pip vs conda 경로 전환
- RTX 5080 드라이버 문제 → NVIDIA 드라이버 업데이트

---

## Gate 2: SimToolReal 에셋 변환 (URDF → USD)

**목표:** Sharpa Hand + KUKA iiwa + tool 에셋을 USD로 변환, Isaac Lab에서 로드 확인

### 2-1. SimToolReal 클론 + 에셋 파악

```bash
git clone https://github.com/tylerlum/simtoolreal.git
# 에셋 위치 확인
ls simtoolreal/assets/urdf/dextoolbench/
```

### 2-2. URDF → USD 변환

```bash
# 실제 사용한 변환 스크립트: scripts/convert_urdf_to_usd.sh
# Isaac Lab의 convert_urdf.py CLI 사용 (UrdfConverter Python API 대신)
bash scripts/convert_urdf_to_usd.sh
```

변환 옵션:
- Robot: `--fix-base` (kinematic chain 보존 위해 `--merge-joints` 미사용)
- Tools/Tables/Environments: `--fix-base --merge-joints` (정적 오브젝트 단순화)

### 2-3. 변환 검증

변환 후 반드시 확인:
- [x] Joint 개수가 원본과 동일 (KUKA 7 + Sharpa 22 = 29 revolute)
- [x] Joint limit 값이 원본 URDF와 일치 (USD deg ↔ URDF rad 변환 확인, 29/29 match)
- [x] Collision mesh가 정상 — USD prim 구조 검증 완료
- [x] Link hierarchy가 보존됨

**변환 결과 (2026-05-11):**

| Category | Count | Status |
|---|---|---|
| Robot (KUKA + Sharpa) | 1 | Converted |
| Tools (6 categories × 2) | 12 | Converted |
| Tables | 6 | 6 converted (bowl_plate 재변환 완료) |
| Environments | 24 | All converted |
| **Total** | **43** | **43 USD** |

**이슈 및 해결:**
- `table_narrow_bowl_plate.urdf`가 YCB 메시(`024_bowl`, `029_plate`)를 참조하나, SimToolReal 레포에 포함되지 않음
- YCB S3에서 메시 다운로드 + CoACD로 collision decomposition 생성 후 재변환 성공
- `scripts/convert_urdf_to_usd.sh`에 `set -e` 누락 → `set -euo pipefail`로 수정

### 주의: Joint 순서 변경

Isaac Lab은 breadth-first, IsaacGym은 depth-first 조인트 순서를 사용.
변환 후 `Articulation.data.joint_names`로 순서를 확인하고, 기존 코드의 인덱싱을 매핑해야 함.

**검증 완료 (2026-05-11):** `scripts/joint_remapping.py`의 매핑 테이블이 URDF kinematic tree의 BFS/DFS 순회와 정확히 일치함을 확인:
- `JOINT_NAMES_ISAACGYM` = URDF DFS 순서 (PASS)
- `JOINT_NAMES_ISAACLAB` = URDF BFS 순서 (PASS)
- Round-trip 매핑 identity (PASS)
- **완료 (2026-05-11):** Gate 4 Phase 1에서 `Articulation.data.joint_names` 29개가 `JOINT_NAMES_ISAACLAB`과 정확히 일치함을 확인

### 실패 시
- URDF 파싱 에러 → URDF 파일 수동 수정 (mesh 경로 등)
- Collision mesh 깨짐 → Isaac Lab의 collision approximation 옵션 조정
- ~~YCB 메시 누락 → YCB S3에서 다운로드 + CoACD decomposition 생성~~ (해결됨)

### 적용된 수정사항
- `isaacsim.asset.importer.urdf`를 Kit config dependencies에 추가
- `set_merge_fixed_ignore_inertia` API mismatch 패치 (Isaac Sim 4.5 vs Isaac Lab v2.3)
- 변환 스크립트 glob 패턴 수정
- `set -euo pipefail` 적용

---

## Gate 3: 로컬 5080 QLoRA 1-step 테스트

**목표:** QLoRA로 5080 16GB에서 fine-tuning 가능 여부 확인

### 3-1. 환경 추가

```bash
# 통합 .venv에서 추가 설치
source .venv/bin/activate
uv pip install bitsandbytes peft
```

### 3-2. QLoRA 1-step 테스트

Gate 6에서 확인한 모델 클래스에 따라 접근 방법이 달라짐:

**경로 A: HF transformers 호환 시**

```python
from transformers import AutoModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
model = AutoModel.from_pretrained("nvidia/GR00T-N1.7-3B", quantization_config=bnb_config)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"))
```

**경로 B: 커스텀 클래스 시**

- GR00T 모델 코드에서 backbone (Eagle vision encoder + LLM) 구조 파악
- Linear 레이어에 직접 PEFT LoRA inject (`peft.inject_adapter_in_model`)
- 4-bit quantization은 `bitsandbytes.nn.Linear4bit`로 수동 교체, 또는 `torchao` 사용

### 통과 기준
- [x] 5080 16GB에서 OOM 없이 1 step forward + backward 완료
- [x] VRAM 사용량 기록: 로드 5.36GB → 학습 5.83GB (batch_size=1)

**QLoRA 테스트 결과 (2026-05-11):**

| 항목 | 값 |
|---|---|
| 경로 | A (HF transformers 호환) |
| 양자화 | 4-bit NF4, double quant, bf16 compute |
| LoRA | r=16, alpha=32, backbone Linear4bit only (216 layers) |
| Trainable params | 17M / 2.56B (0.7%) |
| VRAM (load → train) | 5.36GB → 5.83GB (bs=1) |
| VRAM 추정 (bs별) | bs=2 ~7.4GB, bs=4 ~10.6GB, bs=8 ~17GB (OOM) |
| 16GB 한계 batch size | bs=4~6 (activation ~1.6GB/sample, 나머지 고정) |
| effective batch 확장 | bs=4 + grad_accum=8 → effective 32 가능 |
| Loss (1 step) | 0.5014 |
| Full fine-tuning (비교) | 15.4GB → OOM (A100 필요) |
| 스크립트 | `scripts/groot/qlora_1step_test.py` |

**주요 이슈 및 해결:**
- `deepspeed` import 에러 → uninstall (단일 GPU에서 불필요)
- `beta_dist.sample()` fp16 미지원 → concentration을 float32로 재생성
- `lm_head` (tied embeddings) 양자화 실패 → `llm_int8_skip_modules`에 추가
- action_head 전체 학습 시 OOM → backbone LoRA만 학습 (action_head frozen)

### 실패 시
- ~~OOM → batch_size 줄이기, gradient checkpointing 시도~~ (해결)
- ~~경로 B 통합 난이도가 높으면 → 시간 상한 설정~~ (경로 A로 해결)
- QLoRA 자체가 안 되더라도 A100 full fine-tuning은 정상 진행

---

## Gate 4: SimToolReal 환경 이식

**목표:** SimToolReal의 IsaacGymEnvs 환경을 Isaac Lab `DirectRLEnv`로 변환

### 4-1. 환경 구조 변환

Isaac Lab 환경 구조:
```
scripts/simtoolreal_isaaclab/
├── agents/
│   ├── __init__.py
│   └── rl_games_ppo_cfg.yaml
├── __init__.py
└── simtoolreal_env.py
```

### 4-2. 주요 변환 항목

참고: `docs/isaac/migrating_from_isaacgymenv.rst`

| IsaacGymEnvs | Isaac Lab |
|---|---|
| `VecTask` 상속 | `DirectRLEnv` 상속 |
| `create_sim()` | `_setup_scene()` |
| `gym.acquire_*_tensor()` + `wrap` + `refresh` | `robot.data.joint_pos` 직접 접근 |
| `pre_physics_step()` | `_pre_physics_step()` + `_apply_action()` |
| `compute_observations()` | `_get_observations()` → `{"policy": obs}` |
| `compute_reward()` | `_get_rewards()` → reward tensor 반환 |
| 수동 `reset_buf` + `progress_buf` | `_get_dones()` → `(resets, time_out)` |
| YAML config | `@configclass` Python |
| 쿼터니언 xyzw | **wxyz** |
| 조인트 depth-first | **breadth-first** |

### 4-3. 쿼터니언 변환 체크리스트

SimToolReal 코드에서 orientation을 다루는 모든 위치를 찾아 xyzw → wxyz 변환:
- [ ] Goal pose 정의
- [ ] Reward 계산 (orientation error)
- [ ] Reset 시 initial pose 설정
- [ ] Observation buffer 구성

### 4-4. PhysX 기본값 차이 확인

참고: `docs/isaac/comparing_simulation_isaacgym.rst`

| 파라미터 | IsaacGym 기본값 | Isaac Sim 기본값 |
|---|---|---|
| Angular Damping | 0.0 | 0.05 |
| Max Linear Velocity | 1000 | inf |
| Max Angular Velocity | 64.0 (rad/s) | 5729.58 (deg/s) |
| Max Contact Impulse | 1e32 | inf |

시뮬레이션 동작이 원본과 달라지면 이 기본값 차이를 먼저 의심.

### 통과 기준
- [x] Isaac Lab에서 Sharpa Hand + KUKA iiwa + tool이 로드됨
- [x] 환경 reset이 정상 동작
- [x] Observation/reward 계산이 에러 없이 실행됨
- [x] 랜덤 액션으로 step이 정상 진행됨

### 실패 시
- 조인트 인덱싱 오류 → `joint_names` 출력 후 매핑 테이블 작성
- 시뮬레이션 불안정 → PhysX 기본값을 IsaacGym 값으로 명시적 설정
- 시간 초과 → 핵심 1개 태스크만 먼저 이식, 나머지는 추후

---

## Gate 5: Pretrained Policy 로드 + Rollout

**목표:** SimToolReal pretrained policy를 Isaac Lab 환경에서 로드하여 rollout 확인

**상태:** 블로킹. Policy 로드 성공, goal 달성 0%. 상세: `docs/plan/gate5_blockers.md`

### 5-0. Isaac Sim 업그레이드 (Blocker 1/2 해결)

RTX 5080 (SM_120)에서 Isaac Sim 4.5의 NVRTC 에러로 2개 blocker 발생.
Blocker별로 다른 Isaac Sim 버전이 필요하므로 **multi-venv 전략** 채택:

```bash
# venv-5.1: Blocker 1 (RigidObject PhysX NVRTC) 해결
uv venv --python 3.11 .venv-5.1
source .venv-5.1/bin/activate
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
# Isaac Lab v2.3.x 설치 (release/2.3.0 브랜치)

# venv-6.0: Blocker 2 (Livestream 렌더링 NVRTC) 해결
uv venv --python 3.12 .venv-6.0
source .venv-6.0/bin/activate
# Isaac Sim 6.0 (Early Developer Release) — 소스 빌드 권장
```

### 5-1. Policy 로드

```bash
cd simtoolreal
python download_pretrained_policy.py
```

rl_games checkpoint(.pth)를 Isaac Lab의 rl_games play 스크립트로 실행:

```bash
python scripts/reinforcement_learning/rl_games/play.py \
    --task=SimToolReal-Direct-v0 \
    --num_envs=6 \
    --checkpoint=simtoolreal/pretrained_policy/model.pth
```

### 통과 기준
- [x] Pretrained policy가 로드됨 (weight shape 호환) — LSTM 1024 + MLP [1024,1024,512,512], 140→172 input
- [ ] Procedural tool이 RigidObject로 로드됨 (venv-5.1에서 확인)
- [ ] 6개 tool category 중 최소 1개에서 tool 조작 성공
- [ ] Livestream으로 시각화 확인 (venv-6.0에서 확인)

### 현재 상태 (2026-05-11)
- Convention wrapper 완료 (BFS↔DFS, wxyz↔xyzw, gym↔gymnasium)
- SAPG hack (coef_id=50.0 append) 적용
- Critical fix 6건 적용 (arm control base, hand action scale, joint obs normalize, fingertip order, obs clamping, keypoint offsets)
- Goal-only reset 구현
- **결과: goal 달성 0%** — Blocker 3개 존재 (상세: gate5_blockers.md)

### 실패 시
- Weight shape 불일치 → 조인트 순서 매핑 문제일 가능성 높음. 인덱스 리매핑 적용
- 성능 저하 → PhysX 파라미터 차이 조정, 쿼터니언 컨벤션 재확인
- NVRTC 에러 → Isaac Sim 5.1/6.0 venv에서 재시도
- 완전 실패 → Isaac Lab 환경에서 재학습 필요 (Week 2로 이월)

---

## Gate 6: GR00T N1.7 Inference 확인

**목표:** 동일 Python 3.10 환경에서 GR00T N1.7 inference 확인

Gate 1~5와 병렬 진행 가능. Gate 3 (QLoRA)의 선행 조건 (모델 클래스 확인).

### 6-1. GR00T 설치

```bash
# Gate 1~4와 동일한 .venv 사용
source .venv/bin/activate

git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
uv pip install -e .

# FFmpeg
sudo apt-get install -y ffmpeg
```

### 6-2. Inference 테스트

```bash
python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

### 통과 기준
- [x] Inference 스크립트가 action 벡터 출력 (Avg MSE: 0.020, MAE: 0.077)
- [x] VRAM 사용량 확인: **peak 6.6GB** (16GB 이내, 여유 ~9.7GB)
- [x] 모델 클래스 확인: **HF transformers 호환** (`PreTrainedModel` 상속, `AutoModel.register` 완료) → Gate 3 QLoRA 경로 A 적용 가능

**Inference 결과 (2026-05-11):**

| 항목 | 값 |
|---|---|
| 모델 로딩 시간 | 173.5s |
| Avg inference/step | 87.4ms |
| Peak VRAM | 6.6GB |
| Backbone | Cosmos-Reason2-2B (Qwen3-VL) |
| Action dim | 17 joints |

### 실패 시
- ~~N1.7 접근 불가 → N1.5로 전환~~ (해결: HF gated repo 승인 + `huggingface-cli login`)
- CUDA 버전 불일치 → CUDA 12.8 설치

---

## 실행 순서

```
Day 1: 환경 셋업
├── Isaac Lab 설치 (Gate 1) ✅
├── SimToolReal 클론 + 에셋 파악
├── GR00T 클론 + inference 확인 (Gate 6, 병렬) ✅
└── QLoRA 1-step 테스트 (Gate 3) ✅

Day 2: 에셋 변환 (Gate 2) ✅
├── URDF → USD 변환
├── 변환 검증 (joint 개수, limit, collision)
└── Joint 순서 매핑 테이블 작성

Day 3-4: 환경 이식 (Gate 4) ← 진행 중
├── DirectRLEnv 구조로 환경 코드 변환
├── 쿼터니언 xyzw → wxyz 변환
├── PhysX 기본값 조정
└── 랜덤 액션으로 step 테스트

Day 5: Policy 로드 + 검증 (Gate 5)
├── Pretrained policy rollout
└── 성능 비교 (정성적)

Day 5: 판단
├── Migration 성공 여부 정리
├── 재학습 필요 여부 판단
└── Week 2 계획 수립
```

---

## Week 1 종료 시 판단 기준

| 결과 | 다음 행동 |
|---|---|
| Gate 1~6 모두 통과 | Week 2: 데이터 수집 파이프라인 + GR00T fine-tuning |
| Gate 1~4 통과, Gate 5 실패 (성능 저하) | Isaac Lab에서 재학습 (rl_games PPO). 원본 대비 성능 비교 |
| Gate 4 실패 (이식 난항) | 핵심 1개 태스크에 집중, 나머지 태스크 이식은 추후 |
| Gate 2 실패 (에셋 변환) | Sharpa Hand USD를 수동 제작 or 커뮤니티 에셋 탐색 |
| Gate 6 실패 (GR00T) | N1.5로 전환, 프로젝트 핵심은 유지됨 |

---

## 부수 작업 (게이트와 병행)

- [ ] SimToolReal 코드 읽기: goal pose 시퀀스 정의 방식 파악 (`dextoolbench/`)
- [ ] GR00T custom embodiment 문서: `getting_started/finetune_new_embodiment.md`
- [ ] GR00T modality config 구조: `examples/SO100/so100_config.py` 분석
- [ ] RunPod 예산 산정 (GR00T fine-tuning용, 시뮬레이션은 로컬)

---

## 리스크 로그

| 리스크 | 영향 | 완화 |
|---|---|---|
| DexPBT/SAPG가 Isaac Lab rl_games에서 미호환 | Gate 5 실패 | 표준 PPO로 먼저 시도, SAPG는 추후 |
| Sharpa Hand URDF→USD 변환 시 collision mesh 깨짐 | Gate 2 실패 | Isaac Lab UrdfConverter 옵션 조정, 수동 검증 |
| 조인트 순서 depth→breadth 매핑 오류 | Gate 4-5 전반 | joint_names 기반 명시적 매핑, 하드코딩 인덱스 제거 |
| PhysX 기본값 차이로 시뮬레이션 동작 불일치 | Gate 5 성능 저하 | IsaacGym 값으로 명시적 override |
