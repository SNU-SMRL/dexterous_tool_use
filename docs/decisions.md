# Decision Log

프로젝트 진행 중 시도/검토/폐기한 사항과 그 이유를 기록.
HTML 프로젝트 페이지 업데이트 시 이 문서를 참고하여 반영.

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
