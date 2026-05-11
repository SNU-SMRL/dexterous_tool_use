# Decision Log

프로젝트 진행 중 시도/검토/폐기한 사항과 그 이유를 기록.
HTML 프로젝트 페이지 업데이트 시 이 문서를 참고하여 반영.

---

## 2026-05-11: 프로젝트 방향 전면 전환 — GR00T post-training → UniDex-VLA + Adaptive Impedance Control

**결정:** "SimToolReal rollout → GR00T N1.7 post-training → language-conditioned tool use" 방향을 폐기하고, "UniDex-VLA (frozen) + Adaptive Impedance Control MLP adapter" 방향으로 전면 전환.

**배경:**

1. UniDex-VLA (Tsinghua, CVPR 2026, arXiv 2603.22264) 발견. 50K+ trajectories, 8종 dexterous hand, FAAS 통합 action space, 3D VLA + flow matching으로 language-conditioned dexterous tool use 81% task progress 달성. 코드/모델/데이터 전부 공개. → 기존 "SimToolReal rollout → GR00T post-training" 계획의 핵심 가치가 소멸.

2. UniDex-VLA 태스크 분석 결과, 전부 gentle 태스크 (커피 붓기, 쓸기, 스프레이, 가위, 마우스). Contact-rich interaction (충격, 지속 토크, 저항) 없음. Position-only 제어의 근본적 한계: 자유 공간 → 접촉 → 충격이라는 phase transition을 다룰 수 없음.

3. Force-aware VLA 계열 (ForceVLA, FD-VLA, FAVLA) 조사 — 전부 gripper only. VLM-guided impedance (CompliantVLA-adaptor, OmniVIC, HumanoidVLM) 조사 — 전부 arm only. **Dexterous hand + force-aware + language의 교차점이 비어 있음.**

4. 최종 구조: UniDex-VLA (frozen, position target 생성) + Impedance Adapter MLP (z, a_future, F_contact → adaptive K, D) + Admittance controller (per-joint impedance).

**근거:**
- Gap이 진짜 비어있음: landscape matrix에서 3가지 속성을 동시에 갖는 연구 없음
- GNC 전공 시너지 극대화: impedance/admittance control, gain scheduling이 핵심
- 리스크가 낮은 구조: VLA frozen, 경량 MLP adapter만 학습
- UniDex-VLA 코드 분석 완료: LEAP Hand 16-DoF 공식 지원, internal feature (KV caches) + action chunk (30×82) 추출 가능

**폐기된 것:**
- GR00T N1.7 post-training 전체 계획
- SimToolReal rollout 기반 데이터 생성
- URDF→USD 변환 결과 (Sharpa Hand + KUKA iiwa 43 USD) — LEAP Hand + Franka로 교체
- Gate 1-6 (Week 1 plan) 전체 — 새 Gate 0-3으로 교체

**유지되는 것:**
- Isaac Lab 설치/경험, SM_120/NVRTC 이슈 지식
- API 검증 습관 (CLAUDE.md 규칙)

**핵심 레퍼런스:**
- UniDex-VLA: [arXiv 2603.22264](https://arxiv.org/abs/2603.22264), [GitHub](https://github.com/unidex-ai/UniDex)
- CompliantVLA-adaptor: [arXiv 2601.15541](https://arxiv.org/abs/2601.15541)
- ForceVLA: [arXiv 2505.22159](https://arxiv.org/abs/2505.22159)
- Grasp-to-Act: [arXiv 2602.20466](https://arxiv.org/abs/2602.20466)

---

## 2026-05-11: Isaac Sim multi-venv 전략 채택 — 5.1 + 6.0 병렬 셋업

**결정:** Isaac Sim 4.5 (현재)를 유지하면서, 5.1과 6.0 venv를 병렬로 셋업하여 Gate 5 blocker를 해결한다.

**배경:** RTX 5080 (Blackwell SM_120)에서 Isaac Sim 4.5의 NVRTC 컴파일러가 SM_120을 지원하지 않아 2개 blocker 발생:
1. Procedural tool USD를 `RigidObject`로 로드 시 PhysX NVRTC 에러 (`PhysicsArticulationRootAPI` 존재하는 USD)
2. Livestream (`--livestream 1`) 시 렌더링 파이프라인 NVRTC 에러

**조사 결과:**

| | Isaac Sim 4.5 | 5.1 (GA) | 6.0 (베타) |
|---|---|---|---|
| PhysX SM_120 | X | O (커널 포함) | O |
| 렌더링 SM_120 | X | X (crash 보고) | O (벤치마크 포함) |
| Python | 3.10 | 3.11 | 3.12 |
| Isaac Lab | main (0.54.3) | v2.3.x | v3.0-beta |
| 상태 | GA | GA | Early Developer Release |

**의존성 영향 확인:**
- **GR00T N1.7**: Isaac Sim과 완전 독립 (PyTorch + HuggingFace). 업그레이드 무관.
- **SimToolReal**: 원본은 IsaacGym Preview 4 (Python 3.8). Isaac Lab 포트 없음, SAPG 미포팅. 우리 코드는 이미 Gate 4에서 Isaac Lab DirectRLEnv로 이식 완료. Policy eval은 rl_games inference만 사용.
- **URDF→USD**: Gate 2 변환 결과물 (43 USD)은 Isaac Sim 버전 무관.
- **Isaac Lab 3.0-beta (6.0용)**: API 대변동 (multi-backend, kit-less). Gate 4 env 코드 재수정 필요.

**근거:**
- Blocker 1은 5.1에서 해결 가능 (PhysX SM_120 커널), Blocker 2는 6.0 필요 (렌더링)
- 단일 버전으로는 양쪽 blocker를 동시에 해결 불가
- articulation root 수동 제거로 Blocker 1 우회 시도는 비현실적 — multi-link tool의 joint 구조가 깨짐
- venv 재구성 비용은 낮음 (`uv venv --python 3.11` + 패키지 재설치)
- 6.0은 베타이므로 메인 작업 환경으로 부적합, 검증용으로만 사용

**채택한 전략:**

```
.venv      (Python 3.10, Isaac Sim 4.5)  ← 메인: Blocker 3 디버깅, headless eval
.venv-5.1  (Python 3.11, Isaac Sim 5.1)  ← Blocker 1: procedural tool RigidObject 로드
.venv-6.0  (Python 3.12, Isaac Sim 6.0)  ← Blocker 2: livestream/GUI 시각화
```

**대안 검토:**

| 항목 | 채택한 방법 | 대안 | 판단 |
|---|---|---|---|
| Blocker 1 해결 | Isaac Sim 5.1 venv | USD에서 articulation root 제거 | 5.1이 맞음 — root 제거는 tool joint 구조를 깨뜨림 |
| Blocker 2 해결 | Isaac Sim 6.0 venv | headless + 영상 녹화로 대체 | 6.0 시도 후, 안 되면 영상 녹화 fallback |
| 메인 환경 | 4.5 유지 | 5.1로 전환 | 4.5 유지가 안전 — headless eval은 동작하고, 5.1 전환은 Isaac Lab 버전 변경 수반 |
| 6.0 접근 | 베타 검증용 | GA 대기 | GA 시기 미정이므로 베타라도 먼저 확인 |

**참고 소스:**
- [Isaac Sim 5.1 Requirements (RTX 5080 listed)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Isaac Sim 6.0 Benchmarks (RTX 5080 included)](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/reference_material/benchmarks.html)
- [Isaac Lab Releases](https://github.com/isaac-sim/IsaacLab/releases)
- [RTX 5080 NVRTC Error Forum](https://forums.developer.nvidia.com/t/rtx-5080-on-ubuntu-22-04-nvrtc-error-incorrect-capability-12-0-in-isaac-lab/331905)
- [Isaac Sim 5.1 GUI Crash on Blackwell](https://forums.developer.nvidia.com/t/isaac-sim-5-1-crashes-on-startup-with-rtx-5060-ti-blackwell-sm-120-rtx-scenedb-plugin-crash/366252)

---

## 2026-05-11: Gate 6 + Gate 3 완료 — GR00T N1.7 inference 및 QLoRA 검증

**결정:** GR00T N1.7-3B inference (Gate 5)와 QLoRA 1-step training (Gate 2.5) 모두 RTX 5080 16GB에서 통과.

**Gate 6 결과:**
- Inference: peak VRAM 6.6GB, 87.4ms/step, Avg MSE 0.020
- 모델 구조: `PreTrainedModel` 상속 + `AutoModel.register` → HF transformers 완전 호환
- Backbone: Cosmos-Reason2-2B (Qwen3-VL), Action head: flow-matching DiT

**Gate 3 결과:**
- QLoRA 4-bit (NF4 double quant) + LoRA r=16: VRAM 5.36GB→5.83GB (batch_size=1)
- Trainable: 17M / 2.56B (0.7%), backbone LoRA only
- Activation memory만 batch_size에 비례 → bs=4~6까지 16GB에서 가능 (추정)

**해결한 이슈 4건:**

1. **HuggingFace gated repo 접근** — `nvidia/Cosmos-Reason2-2B`와 `nvidia/GR00T-N1.7-3B` 모두 gated. `huggingface-cli login`으로 Read 토큰 인증 후 해결.

2. **DeepSpeed import 실패 (`CUDA_HOME` 미설정)** — 이 머신에 CUDA toolkit이 없고 PyTorch가 bundled CUDA를 사용. `accelerate`가 무조건 `deepspeed`를 import 시도. `uv pip uninstall deepspeed`로 해결 (단일 GPU에서 불필요).

3. **`beta_dist.sample()` fp16 미지원** — 4-bit 로딩 시 DiT action_head의 Beta distribution concentration 파라미터가 fp16으로 캐스팅됨. PyTorch의 `_sample_dirichlet` CUDA 커널이 fp16을 지원하지 않아 RuntimeError. → `Beta(c1.float(), c0.float())`로 fp32 재생성.

4. **`lm_head` 양자화 실패 (tied embeddings)** — `tie_word_embeddings=True`인 모델에서 `lm_head`가 `Linear4bit`로 감싸지지만 실제 weight는 embedding과 공유되어 `compress_statistics` 속성 누락. → `llm_int8_skip_modules=["action_head", "lm_head"]`로 양자화 제외.

**접근:** 공식 `finetune.sh --max-steps 1`로 full-precision OOM 확인 → `experiment.py:run()`의 model을 4-bit로 교체 → beta_dist/lm_head 이슈 해결.

**대안 검토:**

| 항목 | 채택한 방법 | 대안 | 판단 |
|---|---|---|---|
| QLoRA 접근 | 경로 A (HF `AutoModel` + PEFT) | 경로 B (수동 `inject_adapter_in_model`) | A가 올바름 — 모델이 HF 호환이므로 |
| LoRA 대상 | backbone Linear4bit만 (0.7%) | action_head도 full train (64%) | backbone만이 맞음 — action_head까지 하면 OOM |
| compute dtype | bfloat16 | float16 | bf16이 필수 — DiT의 `beta_dist`가 fp16 미지원 |
| 데이터 로딩 | 공식 pipeline 재사용 | 직접 구성 | 공식 pipeline이 맞음 — 직접 구성은 multi-modal collation이 복잡 |

---

## 2026-05-11: Gate 4 Phase 1-4 완료 — Isaac Lab 환경 이식 (Scene + Reset + Obs + Reward)

**결정:** SimToolReal DirectRLEnv skeleton이 RTX 5080에서 동작 확인. Robot (29 joints) + Table (static) + Object (RigidObject) 로드, reset + random action step 정상.

**해결한 이슈 4건:**

1. **`usd-core` pip 패키지 충돌** — Isaac Sim의 내장 `pxr` 바인딩과 충돌하여 `UsdAPISchemaBase` 에러 발생. 아무 패키지도 의존하지 않는 고아 패키지로 확인, `uv pip uninstall usd-core`로 해결. → **교훈:** 새 패키지 설치 후 기존 환경 smoke test 필수.

2. **Isaac Lab API 오용 4건** — `sim_utils.ImplicitActuatorCfg` (→ `isaaclab.actuators`), `sim_utils.configclass` (→ `isaaclab.utils`), `UsdFileCfg` import 누락, `PhysxCfg(max_depenetration_velocity=...)` (→ `RigidBodyPropertiesCfg` 파라미터). 모두 소스 코드를 확인하지 않고 추정으로 작성한 결과. → **교훈:** CLAUDE.md에 "unfamiliar API는 소스 코드에서 확인" 규칙 추가함.

3. **Table을 RigidObject로 로드 시 NVRTC 에러** — Table USD (URDF에서 `--fix-base`로 변환)에 `PhysicsArticulationRootAPI`가 존재. `RigidObject` + `kinematic_enabled=True` + `articulation_enabled=False` 조합에서 `RigidObjectData` 초기화 중 PyTorch inductor NVRTC 컴파일 실패 (SM_120). → **해법:** 정적 scene 요소는 `RigidObject`가 아니라 `cfg.func()` 직접 스폰 (공식 패턴: `isaaclab_tasks/direct/automate/assembly_env.py`). → **교훈:** NVRTC 에러를 라이브러리 충돌로 오진하여 `LD_LIBRARY_PATH`, `/proc/maps`, 환경변수 등을 조사하느라 시간 소모. 증상이 아니라 아키텍처(정적 body를 RigidObject로 관리) 문제였음. 공식 예제를 먼저 확인했으면 즉시 해결 가능.

4. **Tool USD의 instanced prims에 collision_props 적용 불가** — URDF→USD 변환 시 visuals/collisions가 instanceable로 생성됨. `CollisionPropertiesCfg`는 instanced prim에 적용 불가. USD 자체에 `PhysicsCollisionAPI`가 이미 설정되어 있으므로 spawn cfg에서 `collision_props` 제거로 해결.

**대안 검토:**

| 항목 | 채택한 방법 | 대안 | 판단 |
|---|---|---|---|
| Table 스폰 | `cfg.func()` 정적 스폰 | URDF 재변환 (`--fix-base` 제거) | 채택한 방법이 Isaac Lab 공식 패턴이라 올바름 |
| Task 등록 | `run.py` + `sys.path` | Isaac Lab extension system | 현재는 pragmatic, 추후 extension으로 전환 가능 |
| 파일 구조 | `env_cfg.py` + `env.py` 2파일 | `rewards.py`, `observations.py` 분리 | 2파일이 맞음 (InHandManipulation 예제 패턴) |
| Convention | Isaac Lab native (wxyz, BFS) | IsaacGym 호환 (xyzw, DFS) | native가 맞음, policy wrapper에서 변환 |

---

## 2026-05-11: Gate 2 USD 변환 완료 — YCB 메시 누락 해결

**결정:** YCB 오브젝트(`024_bowl`, `029_plate`)를 직접 다운로드 + CoACD decomposition 생성하여 `table_narrow_bowl_plate` 변환 완료. 43 URDF → 43 USD 1:1 변환 확인.

**배경:** SimToolReal 레포에 `table_narrow_bowl_plate.urdf`가 YCB 메시를 상대경로(`ycb/024_bowl/textured.obj` 등)로 참조하나, 메시 파일이 레포에 포함되지 않음 (라이선스 파일만 존재). 변환 시 "Used null prim" 에러 발생.

**근거:**
- `table_narrow_bowl_plate`은 spatula 태스크(`serve_plate`, `flip_over`)의 필수 테이블
- YCB S3(`ycb-benchmarks.s3-website-us-east-1.amazonaws.com`)에서 tgz 아카이브 다운로드 가능 확인
- CoACD(`pip install coacd`)로 `max_convex_hull=10` decomposition 생성 → URDF 참조와 일치
- 변환 후 USD prim 구조 검증: visual mesh(bowl 8194 verts, plate 8002 verts) + collision decomp 20개 정상 포함
- `scripts/convert_urdf_to_usd.sh`에 `set -e` 누락 → `set -euo pipefail`로 수정

**부수 검증:**
- `scripts/joint_remapping.py`: URDF BFS/DFS 순회와 정확히 일치 확인 (29/29 joints)
- Robot USD: 29 revolute joints, joint limit 29/29 match (rad→deg)
- `scripts/verify_usd_conversion.py`는 URDF 측만 검증 — 파일명 변경 또는 USD 검증 로직 추가 권장

---

## 2026-05-11: 외부 dexterous hand 데이터셋 합치기 폐기 → SimToolReal 단독 사용

**결정:** 외부 dexterous hand/tool-use 데이터셋을 GR00T post-training 데이터에 합치지 않는다. SimToolReal 기존 24 tasks (6 tool categories)의 RL rollout 데이터만 사용.

**배경:** 공개된 dexterous hand 데이터셋 26개를 조사. 실제 다운로드 가능하고 tool use를 포함하는 데이터셋은 AgiBot World(12-dim hand)뿐이며, Sharpa 22-DoF과 action space가 달라 retargeting이 필요. XL-VLA, EgoDex, DexMimicGen, DexArt, DexFuncGrasp, ARCTIC 등은 공개되어 있으나 tool use가 없거나 hand embodiment가 다름. EgoScale(NVIDIA), Grasp-to-Act(UIUC), GR-Dexter(ByteDance), Dex1B(UCSD) 등은 미공개.

**근거:**
- Cross-embodiment retargeting (Allegro 16-DoF, Shadow 24-DoF, 6-DoF → Sharpa 22-DoF)은 그 자체로 리서치 프로젝트 수준의 작업
- SimToolReal은 동일 hand/arm에서 24 tasks × goal pose 시퀀스가 이미 정의되어 있고, pretrained policy rollout이 완전 자동
- 프로젝트 핵심 질문은 "GR00T가 tool use를 language-only로 할 수 있는가"이며, 데이터 다양성이 아님
- 사이드 프로젝트 시간을 데이터 retargeting/변환이 아닌 GR00T post-training 파이프라인 자체에 집중

---

## 2026-05-10: IsaacGym → Isaac Lab 이식 결정

**결정:** IsaacGym 사용 폐기, Isaac Lab 이식을 1차 경로로 변경. Week 1을 migration에 집중 투자.

**배경:** RTX 5080(SM_120, Blackwell)에서 IsaacGym Preview 4가 동작 불가 확인. `libPhysXGpu_64.so`에 SM_80까지만 커널 포함, PTX 미포함으로 JIT 불가. Docker로도 해결 안 됨 (호스트 GPU 아키텍처 문제). A100 클라우드는 가능하지만, deprecated 소프트웨어 의존은 장기적으로 비효율.

**근거:**
- [NVIDIA Forum: IsaacGym + RTX 5080 SM_120 미지원](https://forums.developer.nvidia.com/t/isaac-gym-preview-4-incompatible-with-rtx-5080-blackwell-sm-120-libphysxgpu-64-so-missing-sm-120-kernels/367941)
- [IsaacGymEnvs Issue #220: H100 segfault](https://github.com/isaac-sim/IsaacGymEnvs/issues/220)
- Isaac Lab: SM_120 지원, Python 3.10, 활발히 개발 중 (v2.3)
- 이식 시 SimToolReal + GR00T 전부 Python 3.10 통일

**이전 결정 번복:** 2026-05-09 "Isaac Lab 이식 우선순위 하향" → 하드웨어 제약으로 번복

---

## 2026-05-10: 2단계 학습 전략 채택 (SFT → RL)

**결정:** Step 1: SimToolReal RL policy로 demonstration 수집 → GR00T SFT. Step 2: SFT된 GR00T를 Isaac Lab에서 RL fine-tuning. Step 1 완료 후 Step 2 진행 여부 판단.

**배경:** Behavioral cloning만으로는 teacher(RL policy) 성능의 상한을 넘을 수 없음. 2025년 VLA 분야는 "pretrain → SFT → RL" 패러다임으로 수렴 중 (SimpleVLA-RL, pi_RL, PLD 등). GR00T N1.7은 flow-matching action head를 사용하므로 pi_RL 방식이 원리적으로 적용 가능하나, NVIDIA 공식 RL fine-tuning 지원은 아직 없음 (GitHub Issue #104).

**근거:**
- [SimpleVLA-RL](https://arxiv.org/abs/2509.09674) — GRPO로 VLA RL, LIBERO SOTA
- [pi_RL](https://arxiv.org/abs/2510.25889) — flow-matching VLA용 RL, 57.6%→97.6%
- [PLD](https://arxiv.org/abs/2511.00091) — residual RL + distill, 99% LIBERO

**리스크:**
- DexPBT/SAPG 트레이닝 로직이 Isaac Lab rl_games에서 호환되는지 미확인
- Sharpa Hand URDF → USD 변환 시 collision mesh/joint limit 깨질 가능성

---

## 2026-05-09: GPU 선택

**결정:** RTX 5080 → A100 클라우드

**배경:** GR00T N1.7 fine-tuning은 공식 40GB+ VRAM 필요. RTX 5090(32GB)에서 31GB 사용하며 겨우 성공한 사례 확인. RTX 5080(16GB)은 불가.

**근거:** [GitHub Issue #101](https://github.com/NVIDIA/Isaac-GR00T/issues/101) — RTX 5090 batch_size 8에서 31GB 사용.

---

## 2026-05-09: SimToolReal 핸드 확인

**결정:** 프로젝트 문서의 "Allegro Hand" → "Sharpa Hand 22-DoF"로 전면 수정

**배경:** SimToolReal 논문 본문에 Allegro 언급 없음. 시뮬레이션과 실제 모두 Sharpa Hand 사용. 코드베이스가 DexPBT Kuka Allegro 환경에서 fork됐으나 핸드 에셋은 Sharpa로 교체됨.

**근거:** [SimToolReal paper](https://arxiv.org/html/2602.16863v2), [GitHub](https://github.com/tylerlum/simtoolreal)

---

## 2026-05-09: π0.7 비교 섹션 삭제

**결정:** HTML Section X "GR00T vs π0.7 비교" 삭제

**배경:** π0/π0.7은 주로 평행 그리퍼 위주이며 dexterous hand를 지원하지 않음. 비교 대상이 아닌 모델과 별도 섹션을 할애하는 건 불필요.

**처리:** Section 4 "GR00T 선택 이유"에서 한 줄로 흡수.

---

## 2026-05-09: Isaac Lab 이식 우선순위 하향

**결정:** Isaac Lab 이식을 1차 경로 → optional로 변경. IsaacGym 직접 사용을 동등 옵션으로.

**배경:** SimToolReal은 IsaacGym 전용. 이식에 2-3주 소요 예상 (Sharpa Hand USD 변환, env 래퍼 재작성). 사이드 프로젝트에서 이식 자체에 시간을 쓰는 것은 비효율적. LeRobot 데이터 포맷은 시뮬레이터 독립적이므로 IsaacGym에서 수집해도 GR00T post-training에 문제 없음.

---

## 2026-05-09: DexMachina 소속 수정

**결정:** "Columbia/NVIDIA" → "Stanford/NVIDIA"

**근거:** [arXiv 2505.24853](https://arxiv.org/abs/2505.24853) — Shuran Song 등, Stanford/NVIDIA

---

## 2026-05-09: Data Scaling 실험 우선순위 하향

**결정:** MUST → NICE

**배경:** 5개 스케일(500/1K/2K/5K/10K) × fine-tuning은 클라우드 비용이 큼. "Scaling curve"는 논문 수준 산출물이지 사이드 프로젝트 포트폴리오에는 과함. 2-3개 스케일로 추세만 보면 충분.

---

<!-- 새 항목은 위에 추가 (최신순) -->
