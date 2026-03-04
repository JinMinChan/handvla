# handvla

Allegro hand 기반 MuJoCo 실험 환경에서 VLA action interface를 비교하기 위한 저장소입니다.
현재 메인 비교 축은 아래 3개입니다.

- `full_joint16` (16-DoF 직접 제어)
- `tcp12` (4 fingertip TCP xyz)
- `joint-synergy kD` (16-DoF를 저차원 latent로 압축)

## Quick Start

### 1) 환경 설치

```bash
cd /home/minchan/Downloads/workspace/handvla
conda create -y -n handvla python=3.11
conda activate handvla
pip install -U pip
pip install -r requirements.txt
```

Octo 파인튜닝은 현재 `octoketi` 환경 사용:

```bash
conda activate octoketi
```

### 2) 손 시뮬레이터 실행

```bash
conda activate handvla
cd /home/minchan/Downloads/workspace/handvla
PYTHONPATH=. python scripts/sim/run_allegro_hand.py --side right
```

1920x1080 녹화:

```bash
PYTHONPATH=. python scripts/sim/run_allegro_hand.py --side right --record --record-width 1920 --record-height 1080 --record-fps 60
```

### 3) IK 실험

```bash
PYTHONPATH=. python scripts/sim/finger_ik_experiment.py --side right --trials 3 --seed 7
```

## Repository Layout

```text
handvla/
  env/                      # MuJoCo MJCF, assets, camera utils
  scripts/
    sim/                    # hand viewer, finger IK
    data/                   # mustard collect, dataset collect, OXE convert
    train/                  # Octo finetune
    eval/                   # rollout/evaluation
    research/               # synergy basis/build/convert
  docs/                     # 연구 설계/발표용 문서
  tools/                    # 업로드 전 정리 스크립트
  dataset/                  # 실험 데이터 (git ignore)
  models/                   # 학습 모델 (git ignore)
  codex/logs/               # 로그/대표 영상
```

## Script Map (역할)

### Simulation

- `scripts/sim/run_allegro_hand.py`
  - Allegro hand viewer + slider 제어 + 녹화
- `scripts/sim/finger_ik_experiment.py`
  - 손가락별 랜덤 TCP 타겟 IK 도달성 테스트

### Data

- `scripts/data/collect_mustard_grasp.py`
  - 단일 mustard grasp 실행/검증용
- `scripts/data/collect_mustard_grasp_dataset.py`
  - 성공 에피소드만 raw NPZ로 누적 저장
- `scripts/data/convert_mustard_raw_to_oxe.py`
  - raw NPZ -> OXE(RLDS/TFDS)

### Train / Eval

- `scripts/train/finetune_mustard_octo.py`
  - Octo-base 파인튜닝
- `scripts/eval/rollout_mustard_octo.py`
  - 기본 rollout 평가 (full_joint/tcp12/auto)
- `scripts/eval/rollout_mustard_octo_synergy.py`
  - synergy 전용 rollout
- `scripts/eval/rollout_mustard_octo_tcp12.py`
  - tcp12 디버그 전용(legacy)

### Research Utilities

- `scripts/research/build_joint_synergy_basis.py`
  - full_joint 데이터로 PCA basis 생성
- `scripts/research/convert_full_joint_raw_to_synergy_raw.py`
  - full_joint raw -> synergy raw 변환

## End-to-End Pipeline

### A. Dataset Collection (full_joint 예시)

```bash
conda activate handvla
cd /home/minchan/Downloads/workspace/handvla
PYTHONPATH=. python scripts/data/collect_mustard_grasp_dataset.py \
  --action-interface joint16 \
  --target-episodes 100 \
  --no-viewer \
  --out-dir dataset/mustard_grasp_full_joint_diverse
```

### B. OXE 변환

```bash
PYTHONPATH=. python scripts/data/convert_mustard_raw_to_oxe.py \
  --raw-dir dataset/mustard_grasp_full_joint_diverse/raw \
  --out-dir dataset/mustard_grasp_oxe_full_joint_diverse
```

### C. Octo 파인튜닝

```bash
conda activate octoketi
cd /home/minchan/Downloads/workspace/handvla
PYTHONPATH=. python scripts/train/finetune_mustard_octo.py \
  --data-dir dataset/mustard_grasp_oxe_full_joint_diverse \
  --dataset-name mustard_grasp_oxe \
  --action-dim 16 \
  --save-dir models/mustard_octo_overfit_full_joint_diverse
```

### D. Rollout 평가

```bash
conda activate handvla
cd /home/minchan/Downloads/workspace/handvla
PYTHONPATH=. python scripts/eval/rollout_mustard_octo.py \
  --model-path models/mustard_octo_overfit_full_joint_diverse/run_*/final_model \
  --action-interface joint16 \
  --episodes 10
```

## Synergy-k4 Pipeline

### 1) PCA basis 생성

```bash
PYTHONPATH=. python scripts/research/build_joint_synergy_basis.py \
  --raw-dir dataset/mustard_grasp_full_joint_diverse/raw \
  --k-list 4 \
  --out-dir dataset/synergy_basis_diverse_k236
```

### 2) raw 변환 (16 -> k)

```bash
PYTHONPATH=. python scripts/research/convert_full_joint_raw_to_synergy_raw.py \
  --in-raw-dir dataset/mustard_grasp_full_joint_diverse/raw \
  --basis-path dataset/synergy_basis_diverse_k236/full_joint_pca_k4.npz \
  --out-dir dataset/mustard_grasp_synergy_k4_diverse_k236
```

### 3) OXE 변환

```bash
PYTHONPATH=. python scripts/data/convert_mustard_raw_to_oxe.py \
  --raw-dir dataset/mustard_grasp_synergy_k4_diverse_k236/raw \
  --out-dir dataset/mustard_grasp_oxe_synergy_k4_diverse_k236
```

### 4) 파인튜닝 (k=4)

```bash
conda activate octoketi
cd /home/minchan/Downloads/workspace/handvla
PYTHONPATH=. python scripts/train/finetune_mustard_octo.py \
  --data-dir dataset/mustard_grasp_oxe_synergy_k4_diverse_k236 \
  --dataset-name mustard_grasp_oxe \
  --action-dim 4 \
  --save-dir models/mustard_octo_overfit_synergy_k4_selected
```

### 5) synergy rollout

```bash
conda activate handvla
cd /home/minchan/Downloads/workspace/handvla
PYTHONPATH=. python scripts/eval/rollout_mustard_octo_synergy.py \
  --model-path models/mustard_octo_overfit_synergy_k4_selected/run_*/final_model \
  --basis-path dataset/synergy_basis_diverse_k236/full_joint_pca_k4.npz \
  --episodes 10
```

## Current kept artifacts (minimal)

- Dataset
  - `dataset/mustard_grasp_full_joint_diverse`
  - `dataset/mustard_grasp_oxe_full_joint_diverse`
  - `dataset/mustard_grasp_synergy_k4_diverse_k236`
  - `dataset/mustard_grasp_oxe_synergy_k4_diverse_k236`
  - `dataset/synergy_basis*`
- Models
  - `models/mustard_octo_overfit_full_joint_diverse`
  - `models/mustard_octo_overfit_synergy_k4_selected`
- Representative videos
  - `codex/logs/full_joint_diverse_rollout_demo.mp4`
  - `codex/logs/tcp12_cmd_next_palm_local_rollout_demo_20260303.mp4`
  - `codex/logs/synergy_k4_selected_rollout_demo_20260304.mp4`
  - `codex/logs/finger_ik_260219_164246.mp4`

## Pre-push Cleanup

실행 전 dry-run 권장:

```bash
bash tools/cleanup_before_push.sh
bash tools/prune_logs_representative.sh
```

실제 삭제:

```bash
bash tools/cleanup_before_push.sh --execute
bash tools/prune_logs_representative.sh --execute
```

## Notes

- 기본 실행 커맨드는 `PYTHONPATH=.`를 붙여 실행하세요.
- `dataset/`, `models/`는 `.gitignore`에 포함되어 있어 기본적으로 커밋되지 않습니다.
- 연구 설계 문서는 `docs/`를 참고하세요:
  - `docs/iccas-action-interface-plan.md`
  - `docs/joint-synergy-kD.md`
  - `docs/experiment-flow-for-presentation.md`
