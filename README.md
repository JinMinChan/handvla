# handvla

MuJoCo 기반 Allegro hand / Franka+Allegro manipulation 실험 저장소입니다.
핵심 목적은 VLA에서 hand action interface를 어떻게 설계할지 비교하고, 최종적으로 `Franka TCP 6D + hand synergy 4D` 형태의 end-to-end 제어까지 검증하는 것입니다.

## Current Scope

현재 레포에서 바로 다룰 수 있는 메인 흐름은 3개입니다.

- Allegro hand-only 시뮬레이션과 fingertip IK 검증
- mustard grasp용 hand low-dimensional action 비교 (`full_joint`, `tcp12`, `joint-synergy`)
- Franka + Allegro + mustard pick-and-lift end-to-end VLA (`10D = arm TCP 6D + hand synergy 4D`)

현재 확인된 end-to-end 성공 설정은 아래입니다.

- model: `models/pickandlift_arm_tcp_hand_octo_continue_from_step300_20260308/run_260308_180929/checkpoint_000250`
- rollout setting: `--policy-repeat 10`
- result: `3/3 success`
- references:
  - `codex/logs/json/pickandlift_arm_tcp_hand_octo_continue_step250_repeat10_eval3_20260308.json`
  - `codex/logs/videos/pickandlift_arm_tcp_hand_octo_continue_step250_repeat10_demo_20260308.mp4`

## Environments

기본 실행 환경:

- simulation / data tools: `conda activate handvla`
- Octo fine-tuning: `conda activate octoketi`

설치:

```bash
cd /home/minchan/Downloads/workspace/handvla
conda create -y -n handvla python=3.11
conda activate handvla
pip install -U pip
pip install -r requirements.txt
```

## Repository Layout

```text
handvla/
  env/            MuJoCo MJCF, robot assets, viewer utilities
  scripts/
    sim/          simulation launchers and IK experiments
    data/         dataset collection and raw->OXE conversion
    research/     synergy basis construction and raw conversion helpers
    train/        Octo fine-tuning scripts
    eval/         rollout / checkpoint evaluation scripts
  docs/           project design and presentation notes
  dataset/        generated local datasets (gitignored)
  models/         generated local checkpoints (gitignored)
  codex/logs/     representative logs, images, videos (gitignored)
```

## Quick Start

### Allegro hand viewer

```bash
conda activate handvla
python scripts/sim/run_allegro_hand.py --side right
```

1080p recording:

```bash
python scripts/sim/run_allegro_hand.py \
  --side right \
  --record --record-width 1920 --record-height 1080 --record-fps 60
```

### Finger IK experiment

```bash
conda activate handvla
python scripts/sim/finger_ik_experiment.py --side right --trials 3 --seed 7
```

### Franka + Allegro + mustard scene

```bash
conda activate handvla
python scripts/sim/run_franka_allegro_mustard.py --side right
```

Headless smoke test:

```bash
python scripts/sim/run_franka_allegro_mustard.py --no-viewer --steps 200
```

Pre-grasp IK verification:

```bash
python scripts/sim/run_franka_pregrasp_ik.py --side right
```

## Main Pipelines

### 1. Hand-only low-dimensional action study

Main scripts:

- `scripts/data/collect_mustard_grasp.py`
- `scripts/data/collect_mustard_grasp_dataset.py`
- `scripts/research/build_joint_synergy_basis.py`
- `scripts/research/convert_full_joint_raw_to_synergy_raw.py`
- `scripts/train/finetune_mustard_octo.py`
- `scripts/eval/rollout_mustard_octo.py`
- `scripts/eval/rollout_mustard_octo_synergy.py`
- `scripts/eval/rollout_mustard_octo_tcp12.py`

Typical flow:

```bash
conda activate handvla
python scripts/data/collect_mustard_grasp_dataset.py \
  --target-episodes 100 \
  --no-viewer \
  --out-dir dataset/mustard_grasp_raw

python scripts/research/build_joint_synergy_basis.py \
  --raw-dir dataset/mustard_grasp_raw/raw \
  --k-list 4 \
  --out-dir dataset/synergy_basis

python scripts/research/convert_full_joint_raw_to_synergy_raw.py \
  --in-raw-dir dataset/mustard_grasp_raw/raw \
  --basis-path dataset/synergy_basis/full_joint_pca_k4.npz \
  --out-dir dataset/mustard_grasp_synergy_k4

python scripts/data/convert_mustard_raw_to_oxe.py \
  --raw-dir dataset/mustard_grasp_synergy_k4/raw \
  --out-dir dataset/mustard_grasp_synergy_k4_oxe
```

Training:

```bash
conda activate octoketi
python scripts/train/finetune_mustard_octo.py \
  --data-dir dataset/mustard_grasp_synergy_k4_oxe \
  --dataset-name mustard_grasp_oxe \
  --action-dim 4 \
  --window-size 2 \
  --save-dir models/mustard_octo_synergy_k4
```

Evaluation:

```bash
conda activate handvla
python scripts/eval/rollout_mustard_octo_synergy.py \
  --model-path models/mustard_octo_synergy_k4/run_*/best_model \
  --basis-path dataset/synergy_basis/full_joint_pca_k4.npz \
  --episodes 3
```

### 2. Pick-and-lift hand-only baseline

Arm은 IK로 고정하고 hand만 VLA로 학습하는 기준선입니다.

Main scripts:

- `scripts/data/collect_pickandlift_rlds.py`
- `scripts/research/build_pickandlift_hand_synergy_basis.py`
- `scripts/research/convert_pickandlift_raw_to_hand_synergy_raw.py`
- `scripts/train/finetune_mustard_octo.py`
- `scripts/eval/rollout_pickandlift_hand_octo.py`

### 3. Pick-and-lift end-to-end arm+hand VLA

최종 action space:

- arm: TCP `x, y, z, roll, pitch, yaw`
- hand: synergy latent `k=4`
- total: `10D`

Main scripts:

- `scripts/data/collect_pickandlift_rlds.py`
- `scripts/data/collect_pickandlift_corrective_rlds.py`
- `scripts/research/build_pickandlift_hand_synergy_basis.py`
- `scripts/research/convert_pickandlift_raw_to_arm_tcp_hand_synergy_raw.py`
- `scripts/train/finetune_pickandlift_arm_tcp_hand_octo.py`
- `scripts/eval/rollout_pickandlift_arm_tcp_hand_octo.py`
- `scripts/eval/sweep_pickandlift_arm_tcp_hand_checkpoints.py`

Raw collection:

```bash
conda activate handvla
python scripts/data/collect_pickandlift_rlds.py \
  --target-episodes 20 \
  --max-attempts 25 \
  --capture-hz 5 \
  --no-viewer \
  --out-dir dataset/franka_pickandlift_object6d_low2e_owrap_20_fast5hz
```

Build hand synergy basis:

```bash
python scripts/research/build_pickandlift_hand_synergy_basis.py \
  --raw-dir dataset/franka_pickandlift_object6d_low2e_owrap_20_fast5hz/raw \
  --k 4 \
  --out-dir dataset/pickandlift_synergy_basis
```

Convert to `tcp10` raw:

```bash
python scripts/research/convert_pickandlift_raw_to_arm_tcp_hand_synergy_raw.py \
  --raw-dir dataset/franka_pickandlift_object6d_low2e_owrap_20_fast5hz/raw \
  --basis-path dataset/pickandlift_synergy_basis/pickandlift_hand_pca_k4.npz \
  --out-dir dataset/franka_pickandlift_arm_tcp_hand_synergy_k4 \
  --instruction "grasp the mustard bottle"
```

Convert to OXE:

```bash
python scripts/data/convert_mustard_raw_to_oxe.py \
  --raw-dir dataset/franka_pickandlift_arm_tcp_hand_synergy_k4/raw \
  --out-dir dataset/franka_pickandlift_arm_tcp_hand_synergy_k4_oxe \
  --language-default "grasp the mustard bottle"
```

Train:

```bash
conda activate octoketi
python scripts/train/finetune_pickandlift_arm_tcp_hand_octo.py \
  --data-dir dataset/franka_pickandlift_arm_tcp_hand_synergy_k4_oxe \
  --dataset-name mustard_grasp_oxe \
  --action-dim 10 \
  --window-size 4 \
  --save-dir models/pickandlift_arm_tcp_hand_octo
```

Rollout:

```bash
conda activate handvla
python scripts/eval/rollout_pickandlift_arm_tcp_hand_octo.py \
  --model-path models/pickandlift_arm_tcp_hand_octo/run_*/best_model \
  --basis-path dataset/pickandlift_synergy_basis/pickandlift_hand_pca_k4.npz \
  --episodes 3 \
  --no-viewer
```

Current successful deployment setting:

```bash
conda activate handvla
python scripts/eval/rollout_pickandlift_arm_tcp_hand_octo.py \
  --model-path models/pickandlift_arm_tcp_hand_octo_continue_from_step300_20260308/run_260308_180929/checkpoint_000250 \
  --basis-path dataset/pickandlift_synergy_basis_owrap20_20260306/pickandlift_hand_pca_k4.npz \
  --episodes 3 \
  --no-viewer \
  --seed 0 \
  --policy-repeat 10 \
  --save-json codex/logs/json/pickandlift_arm_tcp_hand_octo_continue_step250_repeat10_eval3_20260308.json
```

Recording:

```bash
python scripts/eval/rollout_pickandlift_arm_tcp_hand_octo.py \
  --model-path models/pickandlift_arm_tcp_hand_octo_continue_from_step300_20260308/run_260308_180929/checkpoint_000250 \
  --basis-path dataset/pickandlift_synergy_basis_owrap20_20260306/pickandlift_hand_pca_k4.npz \
  --episodes 1 \
  --no-viewer \
  --seed 0 \
  --policy-repeat 10 \
  --record --record-width 1920 --record-height 1080 --record-fps 60 \
  --record-path codex/logs/videos/pickandlift_arm_tcp_hand_octo_continue_step250_repeat10_demo_20260308.mp4
```

## Notes

- 이 저장소는 `PYTHONPATH=.` 없이 실행하도록 스크립트를 정리했습니다.
- `dataset/`, `models/`, `codex/`는 로컬 생성물이며 git에는 포함하지 않습니다.
- 현재 end-to-end 10D 결과에서 `policy-repeat`는 rollout 시 정책 호출 주기를 뜻합니다. `control_hz=100`에서 `policy-repeat 10`이면 policy는 초당 10번 호출됩니다.

## Documents

- `docs/joint-synergy-kD.md`
- `docs/experiment-flow-for-presentation.md`
- `docs/iccas-action-interface-plan.md`
