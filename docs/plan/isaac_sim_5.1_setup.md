# Isaac Sim 5.1 + Isaac Lab 2.3.x 설치 가이드

**대상:** Python 3.11, RTX 5080 (SM_120), Ubuntu

## 1. venv 생성

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

## 2. Isaac Sim 5.1

```bash
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

## 3. PyTorch (CUDA 12.8)

```bash
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

## 4. Isaac Lab (release/2.3.0, editable)

```bash
cd IsaacLab
git checkout release/2.3.0
```

**주의: flatdict 빌드 이슈 (uv 전용)**

`uv`의 빌드 격리 환경이 최신 setuptools (82.x)를 사용하는데,
`flatdict==4.0.1`이 `pkg_resources`를 build dependency로 선언하지 않아 빌드 실패.
`pip`에서는 발생하지 않음.

```bash
# setuptools 69.x로 다운그레이드 (pkg_resources 포함 버전)
uv pip install "setuptools<70" wheel
# flatdict를 빌드 격리 없이 먼저 설치
uv pip install flatdict==4.0.1 --no-build-isolation
```

그 후 Isaac Lab 패키지 설치:

```bash
uv pip install -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks -e source/isaaclab_rl
```

## 5. 프로젝트 의존성

```bash
# numpy pinning (Isaac Lab이 numpy<2 요구, rl_games/GR00T이 끌어올릴 수 있음)
uv pip install "numpy<2"

# rl_games fork (SAPG 지원)
uv pip install -e simtoolreal/rl_games
uv pip install "numpy<2"  # rl_games 의존성이 numpy>=2로 올릴 수 있으므로 재pin

# flash-attn (GR00T 의존성) — 소스 빌드에 CUDA toolkit 필요하므로 pre-built wheel 사용
# CUDA_HOME 없는 환경에서는 소스 빌드 실패함
uv pip install flash-attn --no-build-isolation \
    --find-links https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.7.4.post1

# GR00T N1.7
uv pip install -e Isaac-GR00T

# QLoRA 관련
uv pip install bitsandbytes peft
```

## 6. EULA 승인 (최초 1회)

```bash
echo "Y" | python -c "import isaacsim"
```

## 7. 검증

```bash
python -c "
import isaaclab; print(f'Isaac Lab {isaaclab.__version__}')
import isaacsim; print('Isaac Sim OK')
import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
import rl_games; print('rl_games OK')
import gr00t; print('gr00t OK')
import bitsandbytes; print('bitsandbytes OK')
import peft; print('peft OK')
import numpy; print(f'numpy {numpy.__version__}')
"
```

## 검증 결과 (2026-05-11)

```
Isaac Lab 0.48.0
Isaac Sim OK
torch 2.7.1+cu128, CUDA True, GPU NVIDIA GeForce RTX 5080
rl_games OK
gr00t OK
bitsandbytes OK
peft OK
numpy 1.26.4
```
