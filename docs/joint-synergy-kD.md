# Joint-Synergy kD Plan (Allegro Mustard Grasp)

## 0) Goal

`16DoF full_joint` 대비 더 작은 action dimension으로 학습 효율을 높이되, 현재 `tcp12(xyz-only)`에서 발생한 grasp failure(thumb/pose ambiguity)를 피한다.

핵심 전략:

1. `tcp12 + IK` 경로를 주 비교축에서 내린다.
2. `joint-space`에서 저차원화를 수행한다.
3. VLA가 `kD synergy latent`를 예측하고, 실행 시 `16 joint`로 복원한다 (IK 없음).

---

## 0.1 Progress Snapshot (2026-02-27)

완료된 항목:

1. `build_joint_synergy_basis.py` 구현 및 실행 완료
2. `convert_full_joint_raw_to_synergy_raw.py` 구현 및 변환 완료
3. `rollout_mustard_octo_synergy.py` 구현 및 실행 검증 완료
4. `k=4` 기준 synergy raw/OXE/모델/롤아웃 smoke 완료

현재 결과 요약:

1. basis 평가(`k=4,6,8,10`)에서 oracle success 100% (30 episode subset)
2. smallest valid default로 `k=4` 선택
3. synergy-k4 모델(500 step smoke) rollout 3 episode success rate 100%

핵심 artifact:

1. basis summary: `dataset/synergy_basis/full_joint_pca_summary.json`
2. synergy raw summary: `dataset/mustard_grasp_synergy_k4/collection_summary.json`
3. eval summary: `codex/logs/synergy_k4_eval_summary_20260227.json`
4. demo video: `codex/logs/synergy_k4_rollout_demo.mp4`

---

## 0.2 Diverse Benchmark Snapshot (2026-02-28)

완료된 항목:

1. Diverse full_joint raw 수집 완료 (`100` 성공 / `103` 시도)
2. Diverse basis 학습 완료 (`k=4,6,8,10`)
3. Diverse full_joint vs synergy_k4 공정 학습 완료 (둘 다 `3000` step)
4. Diverse 10-episode rollout 비교 및 영상 생성 완료

주요 결과:

1. Diverse basis에서 `k=4` 누적 분산 비율 `~0.992236`
2. Oracle replay success (k=4) `1.0`
3. 10-episode rollout success:
   1. full_joint: `100%`
   2. synergy_k4: `100%`
4. Best contact finger 수 평균:
   1. full_joint: `3.0`
   2. synergy_k4: `4.0`

핵심 artifact:

1. collection summary: `dataset/mustard_grasp_full_joint_diverse/collection_summary.json`
2. basis summary: `dataset/synergy_basis_diverse/full_joint_pca_summary.json`
3. comparison report:
   1. `codex/logs/diverse_joint_synergy_comparison_20260228.json`
   2. `codex/logs/diverse_joint_synergy_comparison_20260228.md`
4. comparison videos:
   1. `codex/logs/full_joint_diverse_rollout_demo.mp4`
   2. `codex/logs/synergy_k4_diverse_rollout_demo.mp4`

---

## 1) Current Facts (Already Verified, 2026-02-26)

### 1.1 What worked

1. `full_joint` 오버피팅 모델은 동일 환경에서 높은 성공률.
2. `finger_ik_experiment.py` 단일 손가락 TCP 도달성은 양호.

### 1.2 What failed

1. `tcp12` VLA rollout은 thumb-only 접촉 패턴 반복.
2. Oracle replay (모델 제거, 저장 action 직접 재생) 결과:
   1. `joint16`: `100/100` success
   2. `tcp12 delta`: `0/100` success
   3. `tcp12 absolute`: `0/100` success
3. Thumb-contact 요구를 끄면 `tcp12` oracle도 성공 가능.

### 1.3 Interpretation

`tcp12(xyz-only)`는 각 손가락의 자세 자유도를 충분히 고정하지 못해 (특히 thumb opposition), contact criterion(thumb required)에서 구조적으로 불리하다.  
즉, 현재 실패는 "저차원화 자체 실패"보다 "저차원화 방식 선택 실패"에 가깝다.

---

## 2) Proposed Interface

## 2.1 Primary: Joint-Synergy kD

1. Joint vector: `q in R^16`
2. Synergy latent: `z in R^k`, `k << 16`
3. Decode:
   1. `q_hat = mu + B z`
   2. `B in R^(16 x k)` (PCA basis or learned linear decoder)
4. Apply:
   1. `q_cmd = clip(q_hat, q_min, q_max)`
   2. `data.ctrl[:16] = q_cmd`

No IK in control loop.

## 2.2 Optional extension (if needed)

`synergy kD + thumb intent 1~2D`:

1. `a = [z, u_thumb]`
2. decode 후 thumb 관절(`thj0/thj1`)에 보정 항을 추가
3. 이 확장은 Phase-2로 두고, 우선 pure synergy로 baseline 확보

---

## 3) Why This Is Better Than tcp12 For This Task

1. `joint16 -> kD -> joint16`은 grasp posture 정보(thumb opposition 포함)를 보존할 수 있다.
2. IK ambiguity가 없어 실행 안정성이 높다.
3. 차원 축소 효과는 유지된다 (`16 -> k`).
4. 논문 메시지 구성이 명확해진다:
   1. naive TCP 저차원은 실패
   2. 구조화된 joint 저차원은 성공/효율 개선

---

## 4) Data and Artifact Design

## 4.1 Inputs

1. Raw full-joint dataset:
   1. `dataset/mustard_grasp_full_joint/raw/episode_*.npz`
2. Existing OXE path for full joint:
   1. `dataset/mustard_grasp_oxe_full_joint/...`

## 4.2 New artifacts

1. Synergy basis:
   1. `dataset/synergy_basis/full_joint_pca_k{K}.npz`
   2. fields:
      1. `mu` `(16,)`
      2. `B` `(16,k)`
      3. `explained_variance_ratio` `(>=k,)`
      4. `joint_names` `(16,)`
      5. `k`, `fit_dataset`, `created_at`
2. Synergy raw dataset:
   1. `dataset/mustard_grasp_synergy_k{K}/raw/episode_*.npz`
   2. `action` dim becomes `k`
   3. metadata:
      1. `action_interface = synergy_kd`
      2. `action_semantics = joint_synergy_latent`
      3. `synergy_basis_path`
3. OXE converted dataset:
   1. `dataset/mustard_grasp_oxe_synergy_k{K}/mustard_grasp_oxe_synergy_k{K}/1.0.0/...`

---

## 5) Implementation Plan (File-Level)

## 5.1 New scripts

1. `build_joint_synergy_basis.py`
   1. load full_joint raw actions
   2. fit PCA
   3. pick `k` by CLI or explained-variance threshold
   4. save basis artifact
   5. print reconstruction metrics

2. `convert_full_joint_raw_to_synergy_raw.py`
   1. read `dataset/mustard_grasp_full_joint/raw/*.npz`
   2. encode each action: `z = B^T (q - mu)` (if orthonormal PCA basis)
   3. save new raw episodes with copied image/state/contact/success
   4. save `collection_summary.json` for synergy dataset

3. `rollout_mustard_octo_synergy.py`
   1. same obs/task pipeline as existing rollout
   2. action dim = `k`
   3. decode latent to `q_cmd`
   4. apply joint control directly
   5. same contact-based success metrics

## 5.2 Existing scripts to reuse

1. `convert_mustard_raw_to_oxe.py`:
   1. 그대로 사용 (action dim 자동 반영)
2. `finetune_mustard_octo.py`:
   1. `--action-dim k`로 실행
3. `collect_mustard_grasp_dataset.py`:
   1. 새 수집 없이 full_joint raw를 기반으로 synergy 변환 우선

---

## 6) Hyperparameter and Ablation Matrix

## 6.1 k candidates

1. `k = 4, 6, 8, 10`
2. 선택 기준:
   1. explained variance
   2. oracle replay success
   3. VLA success/sample efficiency

## 6.2 Baselines

1. `full_joint(16D)` baseline
2. `tcp12 naive` baseline (이미 실패 case 확보)
3. `synergy_kD` (main)
4. optional `synergy_kD + thumb_intent`

## 6.3 Training budget

1. Fair compare:
   1. 동일 step (`20k`) 기준
   2. 동일 seed 최소 3개 (`0,1,2`)
2. Quick smoke:
   1. `500 step` shape/check
   2. `3k step` sanity
   3. `20k step` main result

---

## 7) Debugging Plan (Stage Gates)

## Stage A: Basis sanity

목표: `16 -> k -> 16` 복원이 grasp에 충분한지 확인.

체크:

1. Reconstruction error:
   1. global RMSE (joint rad)
   2. per-joint RMSE
   3. thumb joints RMSE (`thj0~thj3`)
2. explained variance
3. clipping rate after decode (`q_hat` out-of-range ratio)

통과 기준:

1. thumb RMSE가 다른 joint 대비 과도하게 크지 않을 것
2. clipping rate 낮을 것 (권장 < 1%)

## Stage B: Oracle replay on synergy raw

목표: "모델 없이" synergy action 재생만으로 성공 가능한지 확인.

체크:

1. success rate (thumb-required criterion)
2. best_contact_fingers distribution
3. mean/best force range

통과 기준:

1. `>= 95%` success (현재 full_joint oracle과 유사)

실패 시 조치:

1. `k` 증가
2. PCA fit data 정규화 방식 점검
3. thumb weighted PCA 또는 thumb-preserving basis 도입

## Stage C: OXE conversion integrity

목표: 학습 입력 포맷 문제 제거.

체크:

1. `action_dim == k`
2. TFDS features dtype/shape 검증
3. dataset_statistics action mean/std finite 확인

통과 기준:

1. NaN 없음
2. `std`가 0에 가까운 축 없음 (dead dimension 방지)

## Stage D: Train smoke

목표: 학습 코드/shape/runtime 오류 제거.

체크:

1. `500 step` 완주
2. loss 감소 추세
3. checkpoint 저장/로드 정상

통과 기준:

1. 크래시 없음
2. final loss < initial loss

## Stage E: Rollout behavior debug

목표: 실제 closed-loop grasp 동작 확인.

체크:

1. per-step `pred_norm`, `q_delta`
2. finger contact frequency
3. thumb-contact timing
4. 영상 (`1920x1080`) 확인

통과 기준:

1. thumb-only 편향 감소
2. multi-finger 접촉 재현
3. success rate가 tcp12 baseline 대비 유의미 향상

---

## 8) Failure Mode -> Root Cause -> Fix Table

1. 증상: rollout이 거의 안 움직임
   1. 원인: action unnormalization mismatch, wrong `action_dim`
   2. 조치: dataset_statistics key/shape 점검, rollout model path와 dataset_name 일치

2. 증상: 움직이지만 success 0%
   1. 원인: synergy basis가 thumb posture를 충분히 보존 못함
   2. 조치: `k` 증가, thumb-weighted basis, thumb_intent 추가

3. 증상: 학습 loss는 낮은데 rollout 실패
   1. 원인: behavior cloning distribution shift + overfit artifact
   2. 조치: action smoothing sweep, control_repeat sweep, multiple seed eval

4. 증상: 특정 손가락만 반복 접촉
   1. 원인: 데이터 편향 또는 latent 축 해석 편향
   2. 조치: finger-balanced resampling, loss reweighting

5. 증상: 관절 saturate/jitter
   1. 원인: decode clipping 과다
   2. 조치: latent scale clamp, decoder regularization

---

## 9) Metrics and Logging Spec

## 9.1 Core metrics

1. `success_rate` (thumb-required)
2. `success_rate_no_thumb_req` (diagnostic)
3. `best_contacts`, `best_contact_fingers`
4. `thumb_contact_step_ratio`
5. `train_steps_to_target_success`

## 9.2 Efficiency metrics

1. action dim (`16 vs 12 vs k`)
2. wall-clock training time
3. GPU/CPU utilization (가능하면)
4. sample efficiency curve (episodes vs success)

## 9.3 Must-save artifacts

1. eval summary json
2. training loss trace
3. rollout mp4 (`codex/logs/`)
4. basis artifact (`dataset/synergy_basis/...`)

---

## 10) Concrete Execution Order

1. Build basis (`k=4,6,8,10`) and choose initial best candidate by oracle replay.
2. Convert full_joint raw to synergy raw (`k=best`).
3. Convert synergy raw to OXE.
4. Finetune Octo (`action_dim=k`).
5. Rollout evaluation + video.
6. Compare with full_joint and tcp12 on same criteria.
7. If needed, add `thumb_intent` extension and re-run.

---

## 11) Minimum Acceptance Criteria (for "usable low-dim interface")

1. Oracle replay success >= 95% (thumb-required)
2. VLA rollout success is not catastrophically below full_joint baseline
3. action dim reduced from 16 to `k <= 8`
4. training wall-clock or steps-to-reach-threshold better than full_joint

---

## 12) Reporting Frame for Paper/Advisor

권장 메시지:

1. Naive TCP low-dim (`tcp12`)은 grasp-critical posture 정보를 잃어 실패.
2. Joint-synergy low-dim은 posture 정보를 유지하며 차원 축소 가능.
3. 따라서 "low-dim action interface"의 핵심은 단순 차원 축소가 아니라, task-relevant structure를 보존하는 parameterization이다.

---

## 13) Immediate Next Coding Tasks

1. `build_joint_synergy_basis.py` 구현
2. `convert_full_joint_raw_to_synergy_raw.py` 구현
3. `rollout_mustard_octo_synergy.py` 구현
4. README + AGENTS 업데이트 (synergy 실험 표준 명령 포함)
