# ICCAS Plan: Unified Arm-Hand VLA Action Interface Co-Design

## 1) 논문 핵심 질문

- 문제 정의: VLA를 dexterous hand까지 end-to-end로 확장할 때, hand action interface를 어떻게 설계해야 학습 가능성과 제어 성능을 동시에 확보할 수 있는가?
- 핵심 주장: "핸드를 VLA에 붙였다"가 아니라, "같은 VLA/같은 데이터/같은 학습 예산에서 어떤 action interface가 가장 효율적인가"를 정량 비교한다.

## 2) 비교 대상 (고정)

- `full_joint16`: Allegro 16-DoF 직접 예측
- `tcp12_xyz`: 4 fingertip TCP x/y/z (12-DoF)
- `synergy-k`: 16 joint를 PCA 기반 저차원 latent(`k`)로 압축 후 VLA는 `k`만 예측
  - 우선 `k=4`를 메인
  - 보조 실험 `k=2,3,6`

## 3) 공정 비교 원칙

- 동일 scene/object/task: mustard fixed-air grasp
- 동일 success rule: thumb-required + contact/finger/force/stability
- 동일 학습 budget: step 수, batch, optimizer, horizon 최대한 통일
- 동일 rollout protocol: 초기 pose, max steps, smoothing/control repeat 통일
- 동일 report 단위: 10-episode/100-episode 등 명시

## 4) 실험 세트

### E0. 재현 기준선

- 목적: full_joint16 overfit 성능을 재현해 상한선 확보
- 출력: success rate, best contact fingers, mean force, demo video

### E1. Interface 비교 (핵심)

- full_joint16 vs tcp12_xyz vs synergy-k4
- 결과물:
  - 정량 표: 성공률/접촉지표/학습시간/액션차원
  - 정성 영상: 실패 모드(thumb-only, contact miss, jitter)

### E2. Data efficiency

- 데이터 개수 스윕: 예) 25/50/100 episodes
- 질문: 동일 목표 성공률 달성에 필요한 에피소드 수

### E3. Language-conditioned hand intent (확장)

- 단순 `"grasp"` vs 의도 분리 `"pinch"`, `"power grasp"`
- 질문: action interface가 바뀌면 language-to-hand mapping 품질이 달라지는가?

### E4. Robustness (ICCAS 가점)

- object pose jitter/yaw jitter/phase noise 추가
- 질문: 인터페이스별 강건성 차이

## 5) 성공/실패 해석 프레임

- tcp12 실패는 "모델이 못 배움"이 아니라
  - thumb opposition/posture 정보 소실,
  - contact-rich 제약 미표현,
  - cmd semantics 불일치 가능성
  로 분해 분석한다.
- synergy-k는 low-rank prior를 통해 위 문제를 완화하는지 확인한다.

## 6) 논문 Figures / Tables (바로 PPT 전환 가능)

- Figure 1: 문제정의 (7DoF arm + high-DoF hand의 action bottleneck)
- Figure 2: 세 인터페이스 구조도 (joint16 / tcp12 / synergy-k)
- Figure 3: 데이터 파이프라인 (raw -> OXE -> Octo fine-tune -> rollout)
- Figure 4: 대표 rollout 프레임 비교 (성공/실패)
- Table 1: 메인 성능 비교 (success/contact/training cost)
- Table 2: data efficiency (episodes-to-target)
- Table 3: robustness/language ablation

## 7) 저장/산출물 규칙

- 핵심 코드: repo root 또는 `env/` 모듈
- 임시 코드/로그: `codex/`
- 발표용 비교영상: `codex/logs/*_rollout_demo*.mp4`
- 최종 요약: JSON + MD 동시 저장

## 8) 현재까지 확인된 결론 (초기)

- `full_joint16`: 재현/overfit 성능 확보
- `tcp12_xyz`: thumb-required 기준 실패 경향 명확
- `synergy-k`: 현재 mustard 고정 benchmark에서 유망(특히 k=4)

## 9) 다음 실행 우선순위

1. `k=4`를 메인 축으로 full_joint 대비 공정 비교표 완성
2. 데이터량 스윕(E2)으로 "효율"을 수치화
3. 의도형 instruction(E3) 최소 버전 추가
4. ICCAS 제출용 그림/표/영상 패키지 고정

## 10) 기대 기여 문장 (초안)

- "본 연구는 unified arm-hand VLA에서 hand action interface를 실험적으로 분해하여, 단순 차원 축소가 아닌 제어 의미 보존을 만족하는 설계 원칙을 제시한다."
