# Week 1: Gate Plan

## 주의사항

**SimToolReal과 GR00T는 Python 버전이 다릅니다.**
- SimToolReal: **Python 3.8** (IsaacGym Preview 4 요구)
- GR00T N1.7: **Python 3.10**
- 반드시 별도 가상환경 사용

---

## Gate 1: SimToolReal 재현

**목표:** IsaacGym에서 pretrained policy rollout 성공, 기존 태스크 1개 이상 동작 확인

### 1-1. 환경 셋업

```bash
# uv 설치 (없으면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python 3.8 가상환경 생성
uv venv --python 3.8 .venv-simtool
source .venv-simtool/bin/activate

# SimToolReal 클론
git clone https://github.com/tylerlum/simtoolreal.git
cd simtoolreal

# LD_LIBRARY_PATH 설정 (IsaacGym 필요)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  # 또는 venv 경로

# 프로젝트 의존성 설치
uv pip install -e .
```

### 1-2. IsaacGym Preview 4 설치

```bash
# NVIDIA 개발자 포탈에서 다운로드 (계정 필요)
# https://developer.nvidia.com/isaac-gym
# → IsaacGym Preview 4 Package 다운로드

# simtoolreal 디렉토리 바깥에서 압축 해제
cd ..
tar -xzf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
uv pip install -e .

# rl_games 설치
cd ../../simtoolreal/rl_games
uv pip install -e .
```

### 1-3. Pretrained policy 다운로드 + 실행

```bash
cd ../  # simtoolreal 루트
python download_pretrained_policy.py
# → pretrained_policy/config.yaml, pretrained_policy/model.pth 생성

# Interactive evaluation (웹 UI)
python dextoolbench/eval_interactive.py \
  --config-path pretrained_policy/config.yaml \
  --checkpoint-path pretrained_policy/model.pth
# → http://localhost:8080 에서 확인

# GPU 메모리 부족 시
python dextoolbench/eval_interactive.py \
  --config-path pretrained_policy/config.yaml \
  --checkpoint-path pretrained_policy/model.pth \
  --num_envs 6  # 기본값보다 줄임. num_blocks(6)의 배수여야 함
```

### 통과 기준
- [ ] IsaacGym에서 Sharpa Hand + KUKA iiwa가 렌더링됨
- [ ] 6개 tool category 중 최소 1개에서 tool 조작 성공 (goal pose에 도달)
- [ ] rollout 영상/스크린샷 저장

### 실패 시
- IsaacGym 설치 실패 → NVIDIA 포럼/이슈 확인. Preview 4가 현재 CUDA와 호환되는지 체크
- GPU 메모리 부족 → `--num_envs` 줄이기
- 코드+모델이 완전 공개이므로 재현 실패 가능성은 낮음

---

## Gate 2: GR00T N1.7 Inference 확인

**목표:** N1.7 weights 다운로드 + 단일 이미지+언어로 action 출력 확인

### 2-1. 환경 셋업

```bash
# 별도 Python 3.10 환경
uv venv --python 3.10 .venv-groot
source .venv-groot/bin/activate

# GR00T 클론
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
uv sync --python 3.10

# FFmpeg (Ubuntu)
sudo apt-get install -y ffmpeg
```

### 2-2. Inference 테스트

```bash
# weights는 첫 실행 시 자동 다운로드 (~6GB)
uv run python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --traj-ids 1 2 \
    --inference-mode pytorch \
    --action-horizon 8
```

### 통과 기준
- [ ] weights 다운로드 완료 (~6GB)
- [ ] inference 스크립트가 action 벡터 출력 (에러 없음)
- [ ] VRAM 사용량 확인 (16GB 이내여야 함)

### 실패 시
- N1.7 Early Access 접근 불가 → N1.5로 전환 (GA, 안정적)
- CUDA 버전 불일치 → CUDA 12.8 설치 또는 nightly torch

---

## Gate 3: A100 RunPod에서 GR00T Fine-tuning 1-step 테스트

**목표:** OOM 없이 forward + backward 1 step 완료 확인

### 3-1. RunPod 인스턴스 확보

**RunPod** (https://www.runpod.io)
- A100 80GB: ~$1.10/hr (Community Cloud) / ~$1.64/hr (Secure Cloud)
- 가입 시 $5-500 랜덤 크레딧
- 시간 단위 과금, 웹 콘솔에서 바로 실행

```bash
# RunPod에서 A100 80GB 인스턴스 생성 후:
# Template: PyTorch 2.x + CUDA 12.x 선택

# GR00T 설치
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
pip install uv && uv sync --python 3.10
```

### 3-2. Fine-tuning 1-step 테스트

```bash
# 데모 데이터로 fine-tuning 테스트 (max-steps=1로 빠르게 확인)
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 1 \
    --output-dir /tmp/test_finetune \
    --max-steps 1 \
    --global-batch-size 8
```

### 통과 기준
- [ ] OOM 없이 1 step forward + backward 완료
- [ ] VRAM 사용량 기록 (batch_size별)
- [ ] 예상 학습 시간 산출 (1 step 시간 × 2000 steps)

### 확인할 것
- [ ] `--global-batch-size` 8에서 VRAM 얼마?
- [ ] batch_size 줄여야 하나?
- [ ] gradient checkpointing 옵션 있나?
- [ ] LoRA 옵션이 존재하는지 코드 탐색 (`grep -r "lora" gr00t/`)

### 실패 시
- A100 80GB에서 OOM → batch_size 줄이기, gradient checkpointing
- LoRA 미지원 시 → 코드에서 PEFT 통합 직접 구현 (추후)

---

## Gate 4 (Optional): Isaac Lab Dex Hand

**목표:** Isaac Lab에서 Shadow Hand 로드 + 기본 관절 제어 확인

이 게이트는 optional. IsaacGym 직접 사용이 현실적 경로이므로, 시간 여유가 있을 때만 시도.

```bash
# Isaac Lab 설치 (별도 환경 권장)
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html

# Shadow Hand 환경 테스트
python -m isaaclab.scripts.run_env --task Isaac-Shadow-Hand-Over-v0
```

### 통과 기준
- [ ] Shadow Hand가 렌더링되고 관절이 움직임
- [ ] 물체와의 접촉이 정상 작동

---

## 실행 순서 (병렬 가능한 것은 병렬로)

```
Day 1-2: 환경 셋업
├── [로컬] SimToolReal 환경 (Python 3.8 + IsaacGym Preview 4)
├── [로컬] GR00T 환경 (Python 3.10)
└── [RunPod] A100 인스턴스 계정 가입 + 셋업

Day 3: Gate 1 + Gate 2 (로컬)
├── SimToolReal pretrained policy rollout 실행
└── GR00T N1.7 inference 테스트

Day 4: Gate 3 (RunPod)
└── A100에서 GR00T fine-tuning 1-step

Day 5: 판단
├── 게이트 통과 현황 정리
├── 경로 결정: IsaacGym 직접 사용 vs Isaac Lab 이식
└── Week 2 세부 계획 수립
```

---

## Week 1 종료 시 판단 기준

| 결과 | 다음 행동 |
|---|---|
| Gate 1~3 모두 통과 | 정상 진행. Week 2에서 데이터 녹화 파이프라인 구축 |
| Gate 1 실패 (SimToolReal) | IsaacGym 설치 문제일 가능성 높음. 디버깅 후 재시도 |
| Gate 2 실패 (GR00T inference) | N1.5로 전환 (GA, 안정적). 또는 CUDA 버전 문제 해결 |
| Gate 3 실패 (fine-tuning OOM) | batch_size/gradient ckpt 조정. RunPod에서 더 큰 인스턴스 시도 |
| Gate 1+3 통과, Gate 2 실패 | N1.5로 전환해도 프로젝트 핵심은 유지됨 |

---

## 부수 작업 (게이트와 병행)

- [ ] SimToolReal 코드 읽기: goal pose 시퀀스가 어떻게 정의되어 있는지 파악 (`dextoolbench/` 디렉토리)
- [ ] GR00T custom embodiment 문서 읽기: `getting_started/finetune_new_embodiment.md`
- [ ] GR00T modality config 구조 파악: `examples/SO100/so100_config.py` 분석
- [ ] RunPod 예산 산정 (예: 2000 steps × 1step 시간 × ~$1.10/hr)
