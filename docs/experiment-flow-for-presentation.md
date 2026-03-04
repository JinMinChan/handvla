# HandVLA 실험 흐름 정리 (발표 자료용)

## 1) 연구 문제 정의

1. 배경:
   1. 기존 VLA 조작 연구는 주로 low-DoF gripper action 중심.
   2. Allegro hand는 16DoF로 expressive하지만 학습/데이터 비용이 큼.
2. 핵심 질문:
   1. action interface를 저차원화하면 학습 효율과 성능을 동시에 얻을 수 있는가?
   2. 저차원화 방식에 따라 grasp 성공 구조가 어떻게 달라지는가?

---

## 2) 실험 환경 구축

1. MuJoCo Allegro hand-only 환경 구성 (`handvla`).
2. mustard bottle 고정 스폰 grasp scene 구축.
3. contact 기반 성공 판정(thumb contact 포함) 도입.
4. 파지 데이터 수집 파이프라인 구축:
   1. raw NPZ 수집
   2. OXE(RLDS/TFDS) 변환
   3. Octo fine-tuning 및 rollout 검증

---

## 3) 1차 비교: Full Joint vs Naive TCP12

## 3.1 Full Joint (16DoF)

1. action: 16 joint target 직접 예측
2. 결과:
   1. 오버피팅 학습/실행 성공
   2. rollout에서 높은 성공률 재현

## 3.2 Naive TCP12 (4 TCP x xyz)

1. action: fingertip 4개의 위치(12D) 예측 후 IK로 joint 변환
2. 관찰:
   1. 영상에서 thumb 중심 접촉 패턴
   2. multi-finger grasp 실패
3. 정량:
   1. oracle replay 기준 full_joint는 재현 성공
   2. tcp12는 thumb-required 기준에서 실패

---

## 4) 모순 해소: "IK는 되는데 VLA는 왜 실패?"

1. finger IK 테스트는 "목표 TCP를 직접 준 단일 손가락 도달성" 검증.
2. VLA rollout은 "모델이 4손가락 TCP를 동시에 예측"하는 문제.
3. 결론:
   1. IK 자체 문제보다, 모델 출력 분포와 interface 정보 손실이 주요 원인.
   2. `xyz-only tcp`는 grasp-critical posture(특히 thumb opposition) 정보를 충분히 고정하지 못함.

---

## 5) 중간 결론 (발표용 핵심 메시지)

1. 단순 차원 축소(16 -> tcp12)는 성능 저하를 유발할 수 있음.
2. 저차원 interface는 "작다"보다 "task 정보를 보존한다"가 더 중요함.
3. 따라서 naive tcp12는 실패 사례(negative result)로 의미가 있음.

---

## 6) 다음 단계: Joint-Synergy kD

1. 방향 전환:
   1. `tcp12 + IK` 대신 `joint-space 저차원화` 채택.
2. 방법:
   1. full_joint action(16D)에서 synergy basis 학습
   2. VLA는 kD latent 예측
   3. 실행 시 joint로 복원 (`q = mu + Bz`), IK 제거
3. 기대:
   1. 차원 축소 이점 유지
   2. thumb/posture 정보 보존
   3. full_joint 대비 효율 개선 가능성 검증

---

## 7) 최종 실험 구조 (발표 슬라이드용)

1. Baseline A: Full Joint 16D
2. Baseline B: Naive TCP12 (실패 사례)
3. Proposed: Joint-Synergy kD (main)
4. 비교 지표:
   1. success rate (thumb-required)
   2. data efficiency (episode 수 vs 성능)
   3. training time / step efficiency
   4. contact pattern (thumb-only vs multi-finger)

### 현재까지 확보된 실험 팩트 (2026-02-28)

1. Diverse 데이터(100 episodes) 기준 공정 비교에서:
   1. Full joint(16D): 10-episode rollout success 100%
   2. Synergy k4(4D): 10-episode rollout success 100%
2. Contact pattern 차이:
   1. Full joint: best contact finger 평균 3.0
   2. Synergy k4: best contact finger 평균 4.0
3. 해석:
   1. 현재 fixed-scene benchmark에서는 synergy 저차원화가 성능 저하 없이 동작 가능함
   2. 발표 시 "저차원화 성공 사례"로 직접 활용 가능

---

## 8) 발표 시 추천 스토리라인

1. "왜 16D를 줄이고 싶은가?" (동기)
2. "무작정 줄이면 왜 안 되는가?" (tcp12 failure)
3. "어떤 정보가 빠졌는가?" (thumb/posture ambiguity)
4. "그래서 어떤 저차원화를 쓰는가?" (joint-synergy)
5. "결과적으로 무엇을 얻는가?" (효율 + 성능 균형)

---

## 9) 관련 문서 링크

1. 상세 설계: `joint-synergy-kD.md`
2. 협업/결정 기록: `AGENTS.md`
3. 실행 및 명령: `README.md`
