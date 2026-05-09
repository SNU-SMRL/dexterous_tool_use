# 프로젝트 페이지 최종 수정안 (v4.0)

아래는 `project_dexterous_tool.html`의 최종 수정 방향입니다.

> **v4.0 변경사항 (2026-05-09)**
> - 웹서치 기반 사실 검증 반영 (SimToolReal 핸드, DexMachina 소속, π0.7 공개 여부, 5080 VRAM 등)
> - 사이드 프로젝트 성격에 맞게 과도한 연구적 프레이밍 축소
> - RTX 5080 fine-tuning 불가 확인 → 클라우드 GPU를 1차 경로로 전환

### 검증 완료 사항 (웹서치 기반)

| 항목 | 검증 결과 |
|---|---|
| SimToolReal 핸드 | **Sharpa Hand 22-DoF** (Allegro 아님). sim/real 모두 Sharpa. 코드베이스는 DexPBT Allegro에서 fork했으나 핸드 에셋은 Sharpa로 교체됨 |
| SimToolReal 코드 | [GitHub 완전 공개](https://github.com/tylerlum/simtoolreal). pretrained policy 다운로드 가능 (`download_pretrained_policy.py`) |
| SimToolReal 태스크 | 24 tasks, 12 objects, 6 tool categories (Hammer, Marker, Eraser, Brush, Spatula, Screwdriver) |
| GR00T N1.7 | Early Access 공개. [GitHub](https://github.com/NVIDIA/Isaac-GR00T) + [HuggingFace](https://huggingface.co/nvidia/GR00T-N1.7-DROID). 코드 Apache 2.0, weights NVIDIA Open Model License |
| GR00T + EgoScale | EgoScale pretraining(20,854시간)이 N1.7 weights에 내장. 별도 다운로드 불필요. 단 EgoScale 자체 코드/모델은 미공개 ("Coming Soon") |
| GR00T dex hand adapter | **릴리스된 adapter 없음** (UNITREE_G1, LIBERO_PANDA, OXE_WIDOWX만 제공). Custom embodiment 등록 후 직접 학습 필요 |
| GR00T fine-tuning VRAM | 공식 40GB+ 추천. RTX 5090(32GB)에서 31GB 사용하며 겨우 성공. RTX 5080(16GB)은 fine-tuning 불가 → **A100 클라우드 사용 확정** |
| Isaac Lab ↔ GR00T | [IsaacLabEvalTasks](https://github.com/isaac-sim/IsaacLabEvalTasks) repo 존재 (평가용). 학습 데이터 생성은 별도 구축 필요 |
| DexMachina 소속 | **Stanford/NVIDIA** (Columbia 아님). Shuran Song 등, arXiv 2505.24853 |
| π0.7 공개 여부 | π0, π0.5는 [openpi로 오픈소스](https://github.com/Physical-Intelligence/openpi). π0.7은 공개 여부 불명확 (완전 클로즈드는 아님) |
| GR00T N2 | GTC 2026에서 preview 발표. 연말 출시 예정. 현재 코드 없음 |

---

## 1. Section 4 "시스템 설계" — 데이터 파이프라인 재설계

**기존:**
> Isaac Lab에서 RL expert 또는 pretrained/scripted policy를 실행하여 매 스텝 (RGB, language, joint, tool pose)를 녹화. 성공 에피소드 필터링 후 language annotation 자동 생성하여 5K~10K trajectory 확보.

**수정:**

SimToolReal의 pretrained policy를 데이터 소스로 활용하는 파이프라인으로 구체화:

### 데이터 수집 흐름

**SimToolReal pretrained policy (Sharpa Hand 22-DoF + KUKA iiwa 14)**
- IsaacGym에서 pretrained checkpoint 로드 → goal pose 시퀀스 정의 → rollout 실행
- 매 스텝 (RGB, joint state, tool 6D pose, goal pose) 녹화
- 성공 에피소드 필터링 (task completion 기준)

> **참고: Goal pose 시퀀스란?**
> SimToolReal은 goal-conditioned policy. "도구를 이 6D pose로 옮겨라"라는 목표를 입력받아 동작.
> 복합 태스크(e.g. 스패츌라 뒤집기)는 grasp → lift → flip 등 여러 goal pose를 순차 제공해야 함.
> SimToolReal 코드에 기존 태스크의 goal pose가 정의돼 있으므로, 기존 태스크부터 시작하면 이 작업이 크지 않음.

**Language annotation + GR00T 형식 변환**
- 각 성공 궤적에 language instruction 부여 (태스크별 템플릿)
- GR00T post-training 입력 형식 (image, language, action)으로 변환

**핵심:** SimToolReal의 goal pose 모듈은 데이터 생성에만 사용. GR00T는 language → action의 end-to-end 매핑을 학습. 추론 시에는 goal pose 없이 language만으로 동작.

### Isaac Lab 이식 vs IsaacGym 직접 사용

Plan에서는 Isaac Lab 이식을 시도하되, **IsaacGym 직접 사용을 동등한 옵션으로** 둔다.

| 경로 | 장점 | 단점 |
|---|---|---|
| IsaacGym 직접 사용 | SimToolReal이 바로 돌아감. 이식 공수 0 | Isaac Lab 생태계 활용 불가 |
| Isaac Lab 이식 | IsaacLabEvalTasks 등 GR00T 평가 연동 가능 | Sharpa Hand USD 에셋 변환 필요, env 래퍼 재작성, 2-3주 소요 |

사이드 프로젝트 관점에서 IsaacGym 직접 사용이 현실적. Isaac Lab 이식은 여유 있으면 시도.

### 아키텍처 다이어그램 수정

Training과 inference를 분리해서 보여줌:

```
=== Training Pipeline ===

[SimToolReal Policy (Sharpa Hand 22-DoF)]
         |
    IsaacGym에서 rollout (완전 자동)
         |
    (RGB, language, joint, tool pose) × 500~5K+
         |
         v
+--------------------------+
|   GR00T N1.7 (3B)        |  ← post-training (LoRA)
|   + Custom Embodiment     |  ← Sharpa/Shadow Hand adapter 직접 구축
|     Adapter               |
+--------------------------+

=== Inference Pipeline ===

[Language + Camera Image]
         |
+--------------------------+
|   GR00T N1.7 (3B)        |  goal pose 없음
|   System 2: VLM → plan   |  language-only
|   System 1: DiT → action |
+--------------------------+
         |
    dex hand joint actions
```

### 데이터 목표

**500에서 시작, 여유 되면 점진적 확장.**

Rollout이 자동이므로 데이터 자체 생성은 빠르지만, goal pose 시퀀스 정의 + 녹화 파이프라인 구축 + success filtering이 선행돼야 함. 5K~10K는 이상적 목표이지 보장이 아님.

---

## 2. Section 3 "핵심 Gap" — Gap A 수정

**기존:**
> SimToolReal은 goal pose가 주어지면 도구를 다루지만, 인간 비디오에서 추출해야 한다. 자연어 명령에서 자동 생성하는 시스템 부재.

**수정:**
> SimToolReal은 Sharpa Hand 22-DoF로 강력한 object-centric dexterous policy를 제공하지만, (1) goal pose를 외부에서 제공받아야 하며 (인간 비디오 → FoundationPose), (2) language understanding이 없다. 본 프로젝트는 SimToolReal의 데모 데이터로 GR00T N1.7을 post-training하여, goal pose 없이 language만으로 tool use를 수행하는 end-to-end 시스템을 구현한다.

---

## 3. Section 4 "GR00T N1.7을 선택한 이유" — 항목 수정

기존 4개 항목 중 수정 + 추가:

> **EgoScale dexterous pretraining:** GR00T N1.7은 EgoScale 20,854시간 인간 영상 사전학습이 내장되어, 22-DoF hand 동작의 representation을 이미 보유. 다만 릴리스된 embodiment adapter에 dex hand는 없으므로, custom adapter 구축이 필요 (SimToolReal 데이터로 학습하면서 자연스럽게 생성).

> **SimToolReal과의 호환:** SimToolReal은 IsaacGym 기반이며, GR00T는 Isaac Lab 생태계에 속하나 데이터 형식(LeRobot)은 시뮬레이터 독립적. IsaacGym에서 직접 데이터를 수집해도 GR00T post-training에 사용 가능.

기존 항목 수정:
- "Isaac Lab과 같은 NVIDIA 생태계" → "NVIDIA 생태계 + IsaacLabEvalTasks로 Isaac Lab 평가 연동 가능"
- "Inference 16GB+ VRAM (5080 가능), Fine-tuning 40GB+ 추천" → "Inference 16GB+ VRAM. **Fine-tuning 40GB+ 필요 (RTX 5090에서 31GB 사용 확인, RTX 5080은 불가)**"

---

## 4. Section 5 "Week 1 게이트" — 전면 재구성

| 확인 항목 | 구체적 방법 | 통과 기준 | Fallback |
|---|---|---|---|
| GR00T N1.7 weights | HuggingFace 다운로드 → inference 테스트 | 단일 이미지+언어로 action 출력 확인 | N1.5 (GA, 커뮤니티 검증 풍부) |
| SimToolReal 재현 | [레포](https://github.com/tylerlum/simtoolreal) 클론 → `download_pretrained_policy.py` → IsaacGym에서 rollout | 기존 태스크 1개 이상 성공적 조작 | (코드 완전 공개이므로 실패 가능성 낮음) |
| GPU 확보 | 클라우드 A100/L40 인스턴스 테스트 | GR00T LoRA 1 step forward+backward OOM 없음 | 다른 클라우드 (Lambda, RunPod 등) |
| Isaac Lab dex hand (optional) | 설치 → Shadow Hand 로드 → 기본 조작 | 관절 제어 + 물체 접촉 정상 | IsaacGym 직접 사용 |

핵심 변경:
- **"5080 post-training" 게이트 → "클라우드 GPU 확보"로 교체** (5080 fine-tuning 불가 확인됨)
- "RL expert 학습" 제거 → "SimToolReal 재현"
- Isaac Lab은 optional로 강등

---

## 5. Section 6 "실험 설계" — 사이드 프로젝트 관점 조정

| Priority | 실험 | 내용 |
|---|---|---|
| MUST | E2E Demo | language → GR00T → dex hand → task completion. **이것이 프로젝트의 핵심 결과물** |
| MUST | Pretrain 효과 | EgoScale pretrained vs random init. GR00T의 dex pretraining이 tool use에 전이되는지 |
| NICE | Data Scaling | 500 / 2K / 5K 정도로 2-3개 스케일 비교. 포화 지점 확인 |
| NICE | SimToolReal vs GR00T | 동일 태스크에서 SimToolReal(goal pose) vs GR00T(language only). Upper bound 대비 달성률 |
| NICE | Language Sensitivity | "stir clockwise" vs "counterclockwise" 등 |
| STRETCH | Goal Pose Ablation | GR00T에 goal pose를 추가 입력 vs language-only |
| STRETCH | 미학습 도구 zero-shot | 일반화 테스트 |

변경 근거:
- **Data Scaling을 NICE로 하향**: 5개 스케일 × fine-tuning은 클라우드 GPU 비용이 큼. 사이드 프로젝트에서 "scaling curve"는 핵심 산출물이 아님. 2-3개 스케일로 추세만 보면 충분
- **Goal Pose Ablation을 STRETCH로 하향**: 구현 자체는 쉽지만, E2E demo가 먼저

### 평가 지표

- **Task Completion Rate** (핵심)
- **Tool Pose Tracking Error**
- **Language Instruction Following Accuracy**: "clockwise" 지시 시 실제 방향 일치율 등 (language conditioning 검증에 필요)

---

## 6. Section 7 "타임라인" — 재구성

### Month 1 — SimToolReal 재현 + 데이터 확보

| 주차 | 태스크 |
|---|---|
| Wk 1 | **게이트**: GR00T weights inference 확인, SimToolReal 재현 (IsaacGym rollout), 클라우드 GPU 확보 및 GR00T fine-tuning 1-step 테스트 |
| Wk 2 | **데이터 녹화 파이프라인 구축**: SimToolReal rollout → 녹화 → success filtering. 1개 태스크로 시작 |
| Wk 3 | **데이터 수집 + language annotation**: 1-2개 태스크, 500+ trajectories. GR00T 데이터 형식 변환 |
| Wk 4 | **(여유 시) 데이터 확장 또는 Isaac Lab 이식 시도**. 추가 태스크/데이터 확보 |

### Month 2 — GR00T Post-training

| 주차 | 태스크 |
|---|---|
| Wk 5 | **Custom embodiment adapter + 학습 인프라**: Sharpa/Shadow Hand용 modality config 작성. 클라우드 GPU에서 LoRA 셋업 |
| Wk 6 | **첫 학습 실행**: 500 trajectory로 학습. Loss curve 확인, 기본 동작 생성 여부 검증 |
| Wk 7 | **디버깅 + 반복**: 실패 모드 파악, hyperparameter 조정. 데이터 추가 투입 |
| Wk 8 | **초기 E2E 테스트**: language → action → sim에서 실행. 동작 품질 1차 평가 |

### Month 3 — 실험 + 데모

| 주차 | 태스크 |
|---|---|
| Wk 9 | **E2E Demo (MUST)**: 최선의 checkpoint로 task completion 시연. 데모 영상 1차 촬영 |
| Wk 10 | **Pretrain 효과 ablation (MUST)**: EgoScale pretrained vs random init |
| Wk 11 | **NICE 실험**: Data scaling (2-3 스케일), SimToolReal vs GR00T 비교 등 |
| Wk 12 | **결과 정리 + 실패 분석** |

### Month 4 (Buffer) — 산출물 정리

데모 영상 편집, GitHub repo 정리, 포트폴리오 페이지 업데이트.

---

## 7. Section 8 "리스크" — 테이블 수정

**제거:** "RL expert 학습 실패 — 확률: 상"

**수정된 테이블:**

| 리스크 | 확률 | Fallback |
|---|---|---|
| **RTX 5080 fine-tuning 불가** | **확정** | **클라우드 A100/L40 사용 (1차 경로). $1-2/hr** |
| SimToolReal 재현 실패 | 낮 (코드+모델 완전 공개) | 재현 실패 시 scripted policy |
| GR00T custom embodiment adapter 구축 난이도 | 중 | 공식 문서의 custom embodiment 등록 절차 따름. Shadow Hand(Isaac Lab 기본)로 대체 가능 |
| Isaac Lab 이식 실패 | 중 | IsaacGym에서 직접 데이터 수집 (이식 불필요) |
| Sharpa Hand 에셋 미확보 | 중 | Shadow Hand (Isaac Lab 기본 제공) 또는 Allegro로 대체. 단 SimToolReal pretrained policy 직접 사용 불가 |
| GR00T N1.7 Early Access 불안정 | 중 | N1.5 (GA, 커뮤니티 fine-tuning 사례 다수) |
| Contact sim 불안정 | 중 | stir/scoop 등 저충격 태스크 한정 |
| **Language-only로 충분한 성능 안 나옴** | **상** | **E2E demo에서 부분 성공이라도 보여주면 포트폴리오로 충분. 실패 분석 자체가 insight — "왜 안 되는가, goal pose 보조 시 얼마나 개선되는가"를 정량화** |
| 3개월 내 미완성 | 중 | 1개 태스크 E2E demo만이라도 확보 |

---

## 8. Section 9 "기대 산출물" — 사이드 프로젝트 관점

| 산출물 | 내용 |
|---|---|
| Demo 영상 | Language → dexterous tool use 시연 (핵심 산출물) |
| Post-trained GR00T N1.7 | Tool use에 fine-tuned checkpoint |
| Dataset | SimToolReal rollout 기반 500~5K+ trajectories + language annotation |
| 실험 결과 | E2E demo 성공률, pretrain 효과, (가능하면) SimToolReal vs GR00T 비교 |
| GitHub Repo | 코드 + 실험 재현 가이드 |

---

## 9. Portfolio 한 줄 요약 수정

**수정:**
> "SimToolReal의 dexterous tool use 데모 데이터로 GR00T N1.7을 post-training하여, goal pose 없이 language만으로 dexterous tool use를 수행하는 가능성을 Isaac Lab/IsaacGym 시뮬레이션에서 검증했다."

---

## 10. 기존 HTML 추가 수정 사항 (Section 2, 10, X)

### Section 2 핵심 논문 테이블
- DexMachina 그룹: "Columbia/NVIDIA" → **"Stanford/NVIDIA"**

### Section 10 참고 문헌 테이블
- DexMachina 그룹: "Columbia/NVIDIA" → **"Stanford/NVIDIA"**

### Section X "GR00T vs π0.7 비교" — 삭제 권장
π0/π0.7은 dex hand를 지원하지 않으므로 (주로 평행 그리퍼) 본 프로젝트와 비교 대상이 아님. Section 4 "GR00T N1.7을 선택한 이유"에서 한 줄로 언급하면 충분:
> "π0.5/π0.7은 dex hand 데이터 없이 평행 그리퍼 위주이므로 본 프로젝트에 부적합."

별도 섹션(Section X)은 삭제하고, 해당 내용을 Section 4에 흡수.
