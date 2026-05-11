# Decision Log

프로젝트 진행 중 시도/검토/폐기한 사항과 그 이유를 기록.
HTML 프로젝트 페이지 업데이트 시 이 문서를 참고하여 반영.

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
