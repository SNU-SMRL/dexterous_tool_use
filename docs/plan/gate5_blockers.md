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

## 현재 Blocker 2개

### 1. Procedural tool USD가 RigidObject로 로드 불가

- Procedural handle-head URDF → USD 변환 (`--merge-joints`) 시 articulation root 생성
- Isaac Sim 4.5 + RTX 5080 (SM_120)에서 `RigidObjectData` 초기화 중 NVRTC 에러
- claw_hammer.usd는 동작 (articulation root 없음), procedural tool은 실패
- **해법 후보:** `--merge-joints` 없이 변환, 또는 USD에서 articulation root 수동 제거

### 2. Livestream 불가 (시각화)

- `--livestream 1` 시 PyTorch JIT가 Isaac Sim 렌더링의 NVRTC를 사용 → SM_120 미지원으로 실패
- Known issue: Isaac Sim 4.5가 Blackwell GPU 렌더링 미지원
- [Isaac Lab Discussion #2406](https://github.com/isaac-sim/IsaacLab/discussions/2406)
- **해법:** Isaac Sim 5.0+ 업그레이드, 또는 headless + 영상 녹화

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

## 다음 세션에서 할 일

1. Procedural URDF → USD 변환 시 `--merge-joints` 없이 시도 (articulation root 제거)
2. 성공하면 procedural tool로 eval 재시도
3. Isaac Sim 5.0 업그레이드 검토 (livestream + NVRTC 해결)
   - GR00T N1.7은 Isaac Sim 무관 (독립 패키지) → 업그레이드 영향 없음
   - Isaac Lab v2.3과 Isaac Sim 5.0 호환성 확인 필요
4. Medium 미적용 항목 적용
