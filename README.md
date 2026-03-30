# handvla

`handvla`는 MuJoCo 기반 Allegro hand / Franka+Allegro 조작 실험 저장소입니다.
핵심 목적은 **VLA에서 hand action interface를 어떻게 설계할지**를 비교하고, 최종적으로는 **Franka arm + Allegro hand를 하나의 Octo 기반 정책으로 제어**하는 파이프라인을 검증하는 것입니다.

현재 이 레포에서 가장 중요한 최신 결과는 **mustard `wrap_and_lift` official Octo 파이프라인**입니다.

- 최신 기준 모델:
  - `models/mustard_octo_official_wrap500_mix_k4_temporalcontinue_20260325/run_260325_161632/best_rollout_model`
- 최신 대표 영상:
  - `docs/assets/wrap500_temporalcontinue_practical_success_candidate_20260330.mp4`
- 중요한 해석:
  - 이 모델은 **strict benchmark success model은 아닙니다**
  - 하지만 **practical-success candidate**입니다
  - 즉, 실제 closed-loop Octo rollout에서
    - object에 도달하고
    - thumb 포함 full wrap을 만들고
    - 충분히 lift까지 하지만
    - 현재 evaluator의 strict hold continuity 조건을 아직 만족하지 못합니다
  - 현재 reference `10`-episode jitter eval 기준:
    - strict success: `0/10`
    - practical success: `2/10 = 20%`
  - 남은 실패의 큰 비중은 **object 정면이 아니라 옆으로 치우친 pre-grasp alignment**에서 시작됩니다

이 레포는 이 상태를 숨기지 않고 그대로 기록합니다.
즉, 이 결과는 **가짜 성공도 아니고**, **strict benchmark success도 아닙니다**.

---

## 1. 무슨 레포인지

이 레포는 크게 세 흐름을 다룹니다.

1. Allegro hand-only 시뮬레이션과 fingertip IK 검증
2. mustard object 조작에서 hand low-dimensional action interface 비교
3. Franka + Allegro + mustard 조작을 Octo 기반 VLA로 학습/평가하는 end-to-end 파이프라인

현재 최신 관심사는 3번 중에서도 특히:

- **mustard intent wrap task (`wrap_and_lift`)**
- **official Octo dataset/training/evaluation path**
- **goal image conditioned closed-loop VLA inference**

입니다.

이 레포에서 말하는 “official Octo”는 단순 이름만 가져다 쓴 커스텀 정책이 아니라, 실제로:

- `OctoModel.load_pretrained(...)`
- Octo dataset adapter
- Octo observation history wrapper
- Octo temporal ensemble wrapper
- Octo action sampling

을 통해 동작하는 경로를 의미합니다.

---

## 2. 환경 설명

현재 운영 기준 환경은 **`handvla` 하나**입니다.

사용 용도:
- MuJoCo simulation
- 데이터 수집
- raw dataset 가공
- basis 생성
- official Octo fine-tuning
- official Octo rollout / interactive debug

설치 예시:

```bash
cd /home/minchan/Downloads/workspace/handvla
conda create -y -n handvla python=3.11
conda activate handvla
pip install -U pip
pip install -r requirements.txt
```

추가 메모:
- 이 레포는 현재 `handvla`에서 JAX/TensorFlow/Octo 경로까지 한 번에 쓰는 구성을 기준으로 유지합니다.
- 예전 실험 기록에 나오는 `octoketi`는 **과거 분리 운영 흔적**이고, 현재 기준 사용 환경은 아닙니다.
- `requirements.txt`는 기본 시뮬레이션/유틸 패키지 설치 예시입니다. 현재 운영 중인 `handvla`는 여기에 더해 CUDA JAX, TensorFlow, Flax, Optax, TFDS, Transformers, 그리고 Octo-compatible stack이 같이 들어 있는 상태를 기준으로 합니다.

### 로컬 경로 메모
- 로컬에서 별도 Octo source 경로를 쓰고 싶으면 `.octo_src_path`를 둘 수 있습니다.
- 이 파일은 **로컬 전용**이고 저장소에는 포함하지 않습니다.

---

## 3. 각 스크립트 설명

아래는 현재 기준으로 **실제로 중요한 스크립트들**입니다.

### `env/`

#### `env/allegro_hand_trajectories.py`
- Allegro hand의 scripted hand trajectory 정의 파일입니다.
- `thumb_o_wrap` 같은 wrap 계열 손 모양이 여기 들어 있습니다.
- 현재 wrap500 collector와 rollout 해석에서 매우 중요합니다.

#### `env/franka_allegro_mjcf.py`
- Franka + Allegro + mustard MuJoCo scene 조립 파일입니다.
- official wrap / pick-and-lift 평가 환경의 바닥이 되는 scene입니다.

### `octo_data/`

#### `octo_data/mustard.py`
- mustard raw/OXE 데이터를 Octo dataset path에 맞게 연결하는 adapter입니다.
- 특히 goal image를 `task.image_primary`로 넘기는 현재 official wrap pipeline의 핵심입니다.

### `scripts/data/`

#### `scripts/data/collect_mustard_intent_benchmark.py`
- 현재 wrap500 데이터를 만든 핵심 collector입니다.
- `wrap_and_lift`, `push_over`, `hook_and_pull` 같은 mustard intent benchmark task를 수집합니다.
- 최신 collector fixes가 모두 들어 있습니다.
  - break-on-grasp
  - approach early-exit
  - retract-then-upright-z-up lift
  - thumb-oriented wrap trajectory

#### `scripts/data/convert_mustard_raw_to_oxe.py`
- raw episode를 official RLDS/OXE 형식으로 바꾸는 스크립트입니다.
- 현재 wrap500 학습에서 쓰는 goal image와 dataset export가 여기서 정리됩니다.

#### `scripts/data/collect_mustard_intent_dagger_wrap.py`
- wrap task용 DAgger-lite corrective collector입니다.
- 현재 최신 practical-success candidate 자체를 만드는 데 필수는 아니지만, close-phase corrective experiments에 사용했습니다.

#### `scripts/data/collect_pickandlift_rlds.py`
- 기존 pick-and-lift 계열 데이터 수집기입니다.
- wrap500 파이프라인과는 별개의 기존 baseline/reference입니다.

### `scripts/research/`

#### `scripts/research/build_mustard_intent_hand_synergy_basis.py`
- mustard intent raw hand joint data에서 PCA hand synergy basis를 만듭니다.
- wrap500에서는 hand 16D를 `k=4` latent로 줄이는 basis를 여기서 만듭니다.

#### `scripts/research/convert_mustard_intent_raw_to_arm_tcp_hand_synergy_raw.py`
- raw 23D action(`arm7 + hand16`)을 현재 학습용 action interface로 바꾸는 스크립트입니다.
- 최종 action은 `arm TCP 6D + hand synergy kD` 형태입니다.

#### `scripts/research/trim_mustard_intent_raw_prefix.py`
- raw sequence trim/debug 유틸입니다.
- 주 파이프라인의 핵심은 아니고 보조 도구입니다.

### `scripts/train/`

#### `scripts/train/finetune_mustard_octo.py`
- 현재 official Octo mustard 학습의 중심 스크립트입니다.
- 최신 기준으로는:
  - official Octo dataset path 사용
  - `window_size > 1` history 지원
  - `best_val_model` 저장
  - validation-based early stopping
  - training-time rollout probe configurable (`temporal`, jitter 포함)

### `scripts/eval/`

#### `scripts/eval/mustard_intent_gym_env.py`
- official Octo rollout evaluator가 쓰는 gym-style environment입니다.
- 현재 success metric, hold logic, reach metric이 여기 정의됩니다.

#### `scripts/eval/rollout_mustard_intent_octo_official.py`
- official Octo style mustard rollout 엔트리포인트입니다.
- current observation + goal image + language를 받아 실제 Octo model이 action을 sample합니다.
- 현재 wrap500 결과는 이 경로를 기준으로 말합니다.

#### `scripts/eval/run_mustard_intent_octo_interactive.py`
- interactive MuJoCo viewer에서 Octo rollout을 디버그하기 위한 런처입니다.
- `r/p/s/q` 입력으로 reset/pause/single-step/quit이 가능합니다.

#### `scripts/eval/rollout_mustard_intent_octo.py`
- older/custom mustard intent eval debugging 용도입니다.
- current official wrap 결과의 기준 엔트리포인트는 `rollout_mustard_intent_octo_official.py`입니다.

### `scripts/sim/`

#### `scripts/sim/run_allegro_hand.py`
- Allegro hand-only viewer

#### `scripts/sim/finger_ik_experiment.py`
- fingertip IK feasibility test

#### `scripts/sim/run_franka_allegro_mustard.py`
- Franka + Allegro + mustard scene viewer

#### `scripts/sim/run_franka_pregrasp_ik.py`
- pre-grasp IK alignment check

---

## 4. 가장 최근 성공에 가장 가까운 Octo 모델을 만들기까지의 전 과정

이 섹션이 현재 레포에서 제일 중요합니다.

### 4-1. 문제 정의
우리가 풀고 싶었던 문제는 단순 pick-and-lift가 아니라:

- mustard bottle에 접근하고
- hand를 wrap 형태로 감아 쥔 뒤
- 들어 올리는

`wrap_and_lift` task입니다.

핵심 action interface는:
- arm: TCP 6D (`x, y, z, roll, pitch, yaw`)
- hand: synergy latent `k=4`
- 총 `10D`

즉 current official wrap 모델은 **10D arm+hand action**을 예측하는 Octo 정책입니다.

### 4-2. 왜 새로운 데이터를 다시 모았나
이전 official wrap 실험에서는 반복적으로 다음 현상이 있었습니다.

- arm은 object 근처까지 감
- teacher forcing에서는 close branch가 나옴
- rollout에서는 hand가 끝까지 open 쪽에 머무름
- 혹은 thumb-only contact로 끝남

그래서 단순 deterministic clean set만으로는 부족하다고 보고,
**wrap500 recollection**을 새로 만들었습니다.

### 4-3. wrap500 데이터 수집 구성
총 `500`개를 3개 버킷으로 나눠서 수집했습니다.

#### A. clean anchor 100개
- 경로:
  - `dataset/mustard_intent_wrap500_clean100_zupfix_20260324`
- 결과:
  - `100 / 100` 성공
- 특징:
  - spawn jitter 없음
  - arm disturbance 없음
  - 가장 정돈된 정답 trajectory anchor 역할

#### B. bulk randomized 250개
- 경로:
  - `dataset/mustard_intent_wrap500_bulk250_zupfix_20260324`
- 결과:
  - `250 / 271` 성공
- 특징:
  - `spawn_jitter_xy = 0.02`
  - `spawn_yaw_jitter_deg = 10`
  - `settle_steps = 40`
  - `arm_disturb_xyz = (0.01, 0.01, 0.005)`
  - disturb phase = `approach,preshape`
- 목적:
  - approach robustness 확보

#### C. close-focused 150개
- 경로:
  - `dataset/mustard_intent_wrap500_close150_zupfix_20260324`
- 결과:
  - `150 / 157` 성공
- 특징:
  - `spawn_jitter_xy = 0.02`
  - `spawn_yaw_jitter_deg = 10`
  - `settle_steps = 40`
  - `arm_disturb_xyz = (0.006, 0.006, 0.003)`
  - disturb phase = `preshape,close`
- 목적:
  - late close-phase corrective behavior 보강

### 4-4. collector는 그냥 옛날 collector가 아니다
이 500개는 옛날 rough lift collector로 모은 게 아닙니다.

현재 collector에는 다음 수정이 반영되어 있습니다.

1. **break-on-grasp**
- grasp가 안정적으로 잡히면 close 루프를 계속 밀지 않고 바로 탈출

2. **approach early-exit**
- 충분히 정렬되면 긴 approach를 끝까지 채우지 않고 바로 다음 phase로 진행

3. **retract -> upright -> z-up lift**
- 잡은 뒤 바로 불안정하게 말아 올리지 않고,
- 짧게 retract 후 손목 자세를 더 정리하고,
- 그 다음 world-z 기준으로 lift

4. **`thumb_o_wrap` trajectory**
- thumb opposition이 더 자연스럽고 wrap에 유리한 hand scripted trajectory를 사용

즉 wrap500 raw 자체는 현재 우리가 알고 있는 최신 fixed collector 기반입니다.

### 4-5. raw -> k4 basis -> official OXE 변환
수집한 500개 raw는 바로 학습에 쓰지 않았고, 아래 순서로 처리했습니다.

1. raw bundle merge
- `dataset/mustard_intent_wrap500_mix_bundle_20260324`

2. hand synergy basis 생성
- `dataset/mustard_intent_wrap500_mix_k4_basis_20260324/mustard_intent_hand_pca_k4.npz`

3. arm+hand latent raw 변환
- `dataset/mustard_intent_wrap500_mix_k4_20260324`

4. official RLDS/OXE export
- `dataset/mustard_intent_wrap500_mix_k4_20260324_oxe`

여기서 중요한 건 action interface가 최종적으로:
- `arm TCP 6D + hand synergy 4D`
- `action_dim = 10`
이 되도록 맞췄다는 점입니다.

### 4-6. goal image는 어떻게 쓰였나
이건 아주 중요합니다.

현재 파이프라인에서 goal image는 그냥 장식이 아닙니다.

흐름은:
1. exporter가 raw episode 마지막 frame을 `goal_image_primary`로 저장
2. `octo_data/mustard.py`가 그 goal image를 `task.image_primary`로 연결
3. Octo model은 current observation image와 함께 이 task image를 조건으로 사용

즉 학습은:
- current image
- proprio
- language instruction
- goal image
를 입력으로 받아 action을 예측하는 형태입니다.

중요:
- goal image가 reward로 직접 쓰이는 것은 아닙니다
- “최종 이미지랑 비슷하면 성공” 같은 loss는 아닙니다
- 여전히 본질은 imitation / supervised action prediction입니다

### 4-7. official Octo 학습 단계
#### Stage A. 첫 500-wrap official train
- run:
  - `models/mustard_octo_official_wrap500_mix_k4_20260324/run_260324_182938`
- 주요 설정:
  - `pretrained_path = hf://rail-berkeley/octo-base-1.5`
  - `window_size = 4`
  - `batch_size = 8`
  - `num_steps = 5000`
- 결과:
  - `best_train_step = 1127`
  - `best_train_loss ≈ 0.6430`
  - `best_rollout_step = 2000`
  - `best_rollout_score ≈ 217.76`
- 해석:
  - offline fit은 괜찮았지만 rollout은 여전히 좋지 않았음

#### Stage B. validation 중심 continuation
- run:
  - `models/mustard_octo_official_wrap500_mix_k4_continue_20260325/run_260325_151712`
- 시작점:
  - Stage A의 `best_train_model`
- 여기서 바뀐 것:
  - `best_train`보다 `best_val`을 대표 checkpoint로 저장
- 결과:
  - `best_val_step = 4250`
  - `best_val_loss ≈ 0.7844`
  - strict rollout success는 여전히 `0`

#### Stage C. temporal-aligned continuation
- run:
  - `models/mustard_octo_official_wrap500_mix_k4_temporalcontinue_20260325/run_260325_161632`
- 시작점:
  - Stage B의 `best_val_model`
- 여기서 중요한 수정:
  - training-time rollout probe를 더 이상 `rhc` 고정으로 두지 않고
  - `temporal + jitter` 기준으로 checkpoint 선택
- 설정:
  - `rollout_wrapper = temporal`
  - `rollout_episodes = 3`
  - `rollout_spawn_jitter_xy = 0.02`
  - `rollout_spawn_yaw_jitter_deg = 10`
- 결과:
  - `best_train_step = 2103`
  - `best_train_loss ≈ 0.3384`
  - `best_val_step = 1750`
  - `best_val_loss ≈ 0.7574`
  - `best_rollout_step = 2000`
  - `best_rollout_score ≈ 1027.13`

이 단계가 현재 practical-success candidate를 만든 단계입니다.

### 4-8. 최신 모델이 실제로 한 것
최신 reference candidate는:
- `models/mustard_octo_official_wrap500_mix_k4_temporalcontinue_20260325/run_260325_161632/best_rollout_model`

그리고 가장 대표적인 near-success 재현 결과는:
- video:
  - `docs/assets/wrap500_temporalcontinue_practical_success_candidate_20260330.mp4`
- summary:
  - `docs/wrap500_practical_success_candidate_20260330.md`

그 rollout은 실제로:
- `reached = true`
- `approach_min_err ≈ 0.0191 m`
- `best_contacts = 10`
- `best_fingers = ['ff','mf','rf','th']`
- `object_dz_max ≈ 0.1676`

까지 했습니다.

즉,
- arm이 충분히 object front에 감
- hand가 full wrap을 만듦
- bottle을 충분히 들어올림

그런데도 strict metric에서는 `success = false`입니다.

### 4-9. 왜 strict success는 아직 아니냐
현재 evaluator는 아주 엄격합니다.

strict success 조건:
- `reached`
- `lifted`
- `contact_meets`
- 위 조건을 **연속 hold window 동안 유지**

여기서 hold window는:
- `lift_hold_seconds = 1.5`
- `100 Hz`
- 즉 `150 step`

그래서 지금 모델은
- “아무것도 못하는데 가짜로 성공처럼 보이는 것”은 아니고,
- **실제로 거의 다 했지만 hold continuity에서 마지막에 fail**난 상태입니다.

이 레포에서는 이걸 솔직하게 이렇게 부릅니다.

- strict benchmark success: 아직 아님
- practical-success candidate: 맞음

이 구분은 매우 중요합니다.

### 4-10. hand intent / phase auxiliary idea도 있었지만, 최신 모델에는 아직 미적용
우리는 wrap 연구 중간에 **보조 네트워크를 따로 학습해서 phase / close-entry 신호를 만들어 쓰는 아이디어**도 실험했습니다.

구상은 이랬습니다.

1. observation history를 보고
2. 작은 auxiliary 모델이 `close-entry` 또는 `progress`에 해당하는 scalar를 예측
3. 그 신호로 hand correction을 gating
4. 나중에는 이 구조를 `hand intent` 예측으로 확장

즉 현재 task에서의 단순 버전은:
- `obs -> aux_close/progress`
- `aux signal -> hand correction on/off`

장기 버전은:
- `obs -> intent/phase embedding`
- `intent/phase -> policy conditioning 또는 residual expert routing`

입니다.

이 방향을 완전히 버린 것은 아닙니다.
다만 **현재 최신 wrap500 practical-success candidate를 만든 메인라인 모델에는 이 구조를 넣지 않았습니다.**

이유는 두 가지였습니다.

1. 먼저 **pure official Octo baseline**이 clean collector + wrap500 데이터에서 어디까지 가는지 보고 싶었음
2. 최신 failure는 이미 `close-entry` 자체보다 **post-grasp hold continuity**에 더 가까워졌음

즉 현재 최신 모델은:
- auxiliary hand-intent head 없음
- pure official Octo fine-tune + temporal rollout selection

으로 나온 결과이고,
hand intent / phase auxiliary는 **다음 단계 확장 카드**로 문서화해 두는 상태입니다.

### 4-11. 현재 practical success rate와 실패의 주 원인
현재 reference 모델에 대해 같은 deployment 조건으로 `10`-episode jitter eval을 다시 보면:

- 평가 설정:
  - `wrapper = temporal`
  - `conditioning = language_goal`
  - `policy_repeat = 20`
  - `spawn_jitter_xy = 0.02`
  - `spawn_yaw_jitter_deg = 10`
- raw summary:
  - `codex/logs/json/wrap500_temporalcontinue_bestroll_eval10_jitter_20260330.json`
- strict metric:
  - `0 / 10 = 0%`

하지만 우리가 near-success 영상에서 사실상 성공으로 보고 싶은 기준:

- `reached = true`
- `best_fingers`에 `ff,mf,rf,th`가 모두 포함
- `object_dz_max >= 0.08`

으로 다시 계산하면:

- practical summary:
  - `codex/logs/json/wrap500_temporalcontinue_bestroll_eval10_jitter_practical_20260330.json`
- practical success:
  - `2 / 10 = 20%`
- successful episodes:
  - `2`
  - `3`

이 `20%`는 "조금 완화해서 억지로 만든 숫자"라기보다,
현재 모델이 실제로는 이미:
- front approach
- full wrap
- meaningful lift
를 일부 episode에서 달성하고 있다는 뜻입니다.

반대로 남은 실패의 주 원인은 단순히 hand가 못 닫혀서가 아니라,
**grasp를 시도하는 시작 위치가 bottle 중앙선보다 옆으로 치우친 lateral pre-grasp alignment failure**가 크다는 점입니다.

즉 현재 메인 병목은:
- pure close failure
- pure hold failure

둘 중 하나만이 아니라,
먼저 **front-aligned approach를 더 안정화해야 하는 문제**와
그 다음의 **post-grasp hold continuity 문제**
가 함께 남아 있다고 보는 게 맞습니다.

---

## 5. 사용법

아래는 현재 중요한 명령들입니다.

### 5-1. wrap500 수집 + 변환 + 첫 학습 전체 실행
```bash
bash codex/run_wrap500_collect_and_train_20260324.sh
```

이 스크립트는:
- 100 clean
- 250 bulk randomized
- 150 close-focused
를 수집하고,
- bundle merge
- k4 basis 생성
- raw 변환
- OXE export
- official Octo 학습
까지 이어서 실행합니다.

### 5-2. official Octo 학습
```bash
conda activate handvla
python scripts/train/finetune_mustard_octo.py \
  --data-dir dataset/mustard_intent_wrap500_mix_k4_20260324_oxe \
  --save-dir models/mustard_octo_official_wrap500_mix_k4_20260324 \
  --rollout-basis-path dataset/mustard_intent_wrap500_mix_k4_basis_20260324/mustard_intent_hand_pca_k4.npz \
  --rollout-goal-episode dataset/mustard_intent_wrap500_mix_k4_20260324/raw/episode_00000.npz \
  --window-size 4 \
  --batch-size 8
```

### 5-3. validation-based continuation
```bash
conda activate handvla
python scripts/train/finetune_mustard_octo.py \
  --pretrained-path models/mustard_octo_official_wrap500_mix_k4_20260324/run_260324_182938/best_train_model \
  --data-dir dataset/mustard_intent_wrap500_mix_k4_20260324_oxe \
  --save-dir models/mustard_octo_official_wrap500_mix_k4_continue_20260325 \
  --window-size 4 \
  --batch-size 8 \
  --eval-every 250
```

### 5-4. temporal-aligned continuation
```bash
bash codex/run_wrap500_continue_temporalrollout_20260325.sh
```

이 단계는 학습 중 rollout probe도 `temporal + jitter` 기준으로 고릅니다.

### 5-5. official Octo rollout 평가
```bash
conda activate handvla
python scripts/eval/rollout_mustard_intent_octo_official.py \
  --model-path models/mustard_octo_official_wrap500_mix_k4_temporalcontinue_20260325/run_260325_161632/best_rollout_model \
  --basis-path dataset/mustard_intent_wrap500_mix_k4_basis_20260324/mustard_intent_hand_pca_k4.npz \
  --task wrap_and_lift \
  --episodes 10 \
  --wrapper temporal \
  --history-mode per_env_step \
  --conditioning language_goal \
  --resize-mode octo_avg_crop \
  --task-builder create_tasks \
  --goal-episode dataset/mustard_intent_wrap500_mix_k4_20260324/raw/episode_00000.npz \
  --spawn-jitter-xy 0.02 \
  --spawn-yaw-jitter-deg 10 \
  --save-json codex/logs/json/wrap_eval.json
```

### 5-6. practical-success candidate 영상 재현
```bash
conda activate handvla
python scripts/eval/rollout_mustard_intent_octo_official.py \
  --model-path models/mustard_octo_official_wrap500_mix_k4_temporalcontinue_20260325/run_260325_161632/best_rollout_model \
  --basis-path dataset/mustard_intent_wrap500_mix_k4_basis_20260324/mustard_intent_hand_pca_k4.npz \
  --task wrap_and_lift \
  --episodes 1 \
  --seed 2 \
  --policy-repeat 20 \
  --max-policy-steps 60 \
  --wrapper temporal \
  --history-mode per_env_step \
  --conditioning language_goal \
  --resize-mode octo_avg_crop \
  --task-builder create_tasks \
  --goal-episode dataset/mustard_intent_wrap500_mix_k4_20260324/raw/episode_00000.npz \
  --spawn-jitter-xy 0.02 \
  --spawn-yaw-jitter-deg 10 \
  --record \
  --record-dir docs/assets \
  --save-json codex/logs/json/wrap500_temporalcontinue_bestroll_nearsuccess_seed2_20260330.json
```

### 5-7. interactive 디버그
```bash
conda activate handvla
python scripts/eval/run_mustard_intent_octo_interactive.py
```

기본 키:
- `r`: reset
- `p`: pause/resume
- `s`: single-step
- `q`: quit

---

## 참고 문서

- current result summary:
  - `docs/wrap500_practical_success_candidate_20260330.md`
- presentation timeline:
  - `docs/experiment-flow-for-presentation.md`
- action-interface design notes:
  - `docs/joint-synergy-kD.md`
- broader project plan:
  - `docs/iccas-action-interface-plan.md`

---

## 마지막 한 줄 정리
현재 `handvla`의 최신 wrap500 결과는:

- **official Octo VLA closed-loop rollout**로,
- **가짜 성공이 아니라 실제 practical near-success**를 만들었고,
- 남은 병목은 **post-grasp hold continuity**입니다.

이 레포는 그 상태를 그대로 재현 가능하게 정리하는 것을 목표로 유지합니다.
