# Language-Conditioned Dexterous Tool Use via GR00T N1.7

SimToolReal(Stanford, RSS 2026)의 dexterous tool use 데모 데이터로 GR00T N1.7을 post-training하여, goal pose 없이 language만으로 dexterous tool use를 수행하는 가능성을 시뮬레이션에서 검증하는 사이드 프로젝트. EgoScale 20K시간 사전학습이 내장된 GR00T N1.7의 dexterous manipulation prior가 contact-rich tool use로 전이되는지를 실험적으로 확인한다.

**[Project Page](https://snu-smrl.github.io/dexterous_tool_use/)**

## Stack

| Component | Detail |
|---|---|
| Foundation Model | GR00T N1.7 (3B, EgoScale pretrained) |
| Data Source | SimToolReal pretrained policy (Sharpa Hand 22-DoF) |
| Simulator | IsaacGym (+ Isaac Lab optional) |
| Training | A100 Cloud, LoRA |

## Progress

| Phase | Status |
|---|---|
| Week 1: Gate (SimToolReal 재현, GR00T inference, A100 셋업) | Not Started |
| Week 2-4: 데이터 수집 + annotation | Not Started |
| Week 5-8: GR00T Post-training | Not Started |
| Week 9-12: 실험 + E2E Demo | Not Started |
| Week 13-16: 산출물 정리 | Not Started |

## Structure

```
configs/embodiment/   # GR00T custom embodiment configs
scripts/
  data_collection/    # SimToolReal rollout + recording
  data_processing/    # LeRobot format conversion, language annotation
  training/           # GR00T fine-tuning
  evaluation/         # Eval, demo recording
assets/               # Robot URDF/USD, tool meshes
docs/                 # Plans, decision log
```
