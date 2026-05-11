# Gate 5: Pretrained Policy Rollout — Blockers & Status

**상태:** 블로킹. Pretrained policy 로드 성공, 하지만 goal 달성 0%.

## 해결된 것

- [x] Forked rl_games 설치 (SAPG `extra_params` + `coef_cond` 지원)
- [x] Convention wrapper (BFS↔DFS joints, wxyz↔xyzw quats, gym↔gymnasium 인터페이스)
- [x] SAPG hack: obs에 50.0 append (coef_id)
- [x] Checkpoint 로드 성공 (LSTM 1024 + MLP [1024,1024,512,512], 140→172 input)
- [x] eval.py: SimToolReal 지표 (goal_pct = successes/50 × 100%)
- [x] Goal-only reset 구현 (goal 달성 시 환경 유지, goal만 재샘플링)

## Critical Fix 적용 (비교 감사에서 발견)

1. **Arm control base**: `joint_pos` → `prev_targets` (원본 패턴)
2. **Hand action scale**: `scale_transform` → `unscale_transform` (방향 반전 수정)
3. **Joint obs normalize**: `unscale_transform` → `scale_transform` (방향 반전 수정)
4. **Fingertip order**: [thumb,index,middle,ring,pinky] → [index,middle,ring,thumb,pinky]
5. **Obs clamping**: [-10, 10] clamp 추가
6. **Keypoint offsets**: 원본과 일치하도록 수정, per-env object_scale 반영

## 현재 Blocker 3개

### 1. Procedural tool USD가 RigidObject로 로드 불가

- Procedural handle-head URDF → USD 변환 (`--merge-joints`) 시 articulation root 생성
- Isaac Sim 4.5 + RTX 5080 (SM_120)에서 `RigidObjectData` 초기화 중 NVRTC 에러
- claw_hammer.usd는 동작 (articulation root 없음), procedural tool은 실패
- articulation root 수동 제거는 비현실적 — 실제 multi-link tool의 joint 구조가 깨짐
- **해법:** Isaac Sim 5.1 업그레이드 (PhysX에 SM_120 커널 포함, 아래 호환성 분석 참조)

### 2. Livestream 불가 (시각화)

- `--livestream 1` 시 PyTorch JIT가 Isaac Sim 렌더링의 NVRTC를 사용 → SM_120 미지원으로 실패
- Known issue: Isaac Sim 4.5가 Blackwell GPU 렌더링 미지원
- [Isaac Lab Discussion #2406](https://github.com/isaac-sim/IsaacLab/discussions/2406)
- Isaac Sim 5.1에서도 렌더링 NVRTC 미해결 (GUI crash, TiledCamera hang 보고)
- **해법:** Isaac Sim 6.0 업그레이드 (공식 RTX 5080 벤치마크 포함, 렌더링 지원)

### 3. claw_hammer에서도 goal 달성 0%

- Critical fix 적용 후에도 0% (이전에 full-reset에서는 가끔 reward 1103.7 나옴)
- 원인 후보:
  - Object가 다름 (claw_hammer mesh ≠ procedural box+cylinder)
  - object_scale이 (1,1,1)로 틀림 (procedural은 (6.5, 0.8, 0.4) 등)
  - 추가 미식별 불일치 존재 가능

## 비교 감사 결과 (Medium 우선순위, 미적용)

- Goal rotation이 항상 identity (원본은 random quaternion)
- Fixed-size keypoint reward 미구현 (pretrained config에서 사용)
- Target volume bounds 약간 다름 (pretrained config vs 우리 cfg)
- Delta goal sampling 미구현 (절대 위치만)

## Isaac Sim 업그레이드 호환성 분석 (2026-05-11)

### 버전별 SM_120 (RTX 5080) 지원

| | Isaac Sim 4.5 (현재) | Isaac Sim 5.1 | Isaac Sim 6.0 |
|---|---|---|---|
| **PhysX (headless 시뮬)** | SM_120 미지원 | SM_120 커널 포함 | 지원 |
| **렌더링 (GUI/livestream)** | SM_120 미지원 | 미해결 (crash 보고) | RTX 5080 벤치마크 포함 |
| **릴리즈 상태** | GA | GA | Early Developer Release (베타) |
| **Python** | 3.10 | **3.11** | **3.12** |
| **Isaac Lab** | main (0.54.3) | v2.3.x | v3.0-beta (API 대변동) |
| **설치** | pip | pip (`--extra-index-url https://pypi.nvidia.com`) | 소스 빌드 권장 |

### Blocker별 해결 매핑

| Blocker | Isaac Sim 4.5 | 5.1 | 6.0 |
|---|---|---|---|
| 1. RigidObject NVRTC (PhysX) | X | **해결 가능성 높음** | 해결 |
| 2. Livestream (렌더링) | X | X | **해결 가능성 높음** |
| 3. goal 달성 0% | 미해결 | 미해결 | 미해결 |

### 의존성 영향

- **GR00T N1.7**: Isaac Sim과 **완전 독립** (PyTorch + HuggingFace). 업그레이드 영향 없음.
- **SimToolReal pretrained policy**: rl_games inference만 사용. Isaac Lab API 변경 시 convention wrapper 수정 필요.
- **URDF→USD**: Gate 2 변환 결과물(43 USD)은 Isaac Sim 버전 무관, 재변환 불필요.
- **Python 버전**: 5.1은 3.11, 6.0은 3.12 → 새 venv 필수. GR00T은 별도 venv 또는 PyTorch 버전 주의.

### venv 전략

```
venv-4.5  (Python 3.10, 현재)  ← Blocker 3 디버깅, headless eval 계속
venv-5.1  (Python 3.11)        ← Blocker 1 해결 (procedural tool RigidObject 로드)
venv-6.0  (Python 3.12)        ← Blocker 2 해결 (livestream/GUI 시각화)
```

## 다음 세션에서 할 일

1. **venv-5.1 셋업**: Python 3.11 + Isaac Sim 5.1 + Isaac Lab 2.3.x
   - Procedural tool USD가 RigidObject로 로드되는지 확인 (Blocker 1)
2. **venv-6.0 셋업**: Python 3.12 + Isaac Sim 6.0 (베타)
   - Livestream/GUI 동작 여부 확인 (Blocker 2)
3. Blocker 3 (goal 0%) 디버깅 계속 — Isaac Sim 버전 무관
   - Medium 미적용 항목 적용 (random goal rotation, fixed-size keypoint reward 등)
   - object_scale 불일치 수정
