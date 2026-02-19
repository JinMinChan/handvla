# handvla (Allegro hand-only starter)

MuJoCo에서 Allegro hand만 띄워서 오른쪽 컨트롤 패널로 조인트를 직접 조절하는 최소 환경입니다.

## 1) 설치

```bash
cd /home/minchan/Downloads/workspace/handvla
conda create -y -n handvla python=3.11
conda activate handvla
pip install -U pip
pip install -r requirements.txt
```

## 2) 실행

오른손 Allegro hand:

```bash
conda activate handvla
python run_allegro_hand.py --side right
```

왼손 Allegro hand:

```bash
conda activate handvla
python run_allegro_hand.py --side left
```

1920x1080 동영상 녹화:

```bash
conda activate handvla
python run_allegro_hand.py --side right --record
```

- 기본 저장 경로: `codex/logs/allegro_hand_<timestamp>.mp4`
- 커스텀 저장 경로/해상도/FPS 예시:

```bash
python run_allegro_hand.py --record --record-path codex/logs/my_run.mp4 --record-width 1920 --record-height 1080 --record-fps 60
```

## 참고

- UI 오른쪽 패널에서 actuator slider(ffa0~tha3)를 직접 조절할 수 있습니다.
- 모델 파일은 `env/assets/wonik_allegro/` 아래에 있습니다.
- 배경/조명 톤은 `keti_dual_arm_VLA/env/assets/scene.xml` 스타일을 따릅니다.
- 초기 자세는 손목이 아래, 손이 위쪽으로 뻗은 형태입니다.
- 각 fingertip(흰색 tip)에서 손바닥 방향으로 약간 물린 위치에 TCP 마커(빨간 구)가 추가되어 있습니다.
- 실험용 코드/로그/임시 파일은 `codex/` 폴더에서 작업합니다.

## IK 가능성 실험 (랜덤 reachable target)

각 손가락별로 랜덤 타겟을 만들고(TCP가 실제로 도달 가능한 위치만 샘플링), Jacobian 기반 IK로 해당 타겟에 수렴 가능한지 측정합니다.

```bash
conda activate handvla
cd /home/minchan/Downloads/workspace/handvla
PYTHONPATH=. python finger_ik_experiment.py --side right --trials 3 --seed 7
```

- 위 명령은 기본적으로 viewer를 열어서 IK 과정을 보여줍니다.
- 기본 trial 수는 손가락당 3회입니다.
- 기본 viewer 업데이트 속도는 약 20Hz이며, `--viewer-step`으로 조절할 수 있습니다.
- 현재 테스트 중인 손가락의 초록 target 구만 보이도록 처리되어 있습니다.
- viewer 없이 수치 결과만 보려면:

```bash
PYTHONPATH=. python finger_ik_experiment.py --side right --trials 3 --seed 7 --no-viewer
```

JSON 로그 저장 예시:

```bash
PYTHONPATH=. python finger_ik_experiment.py --save-json codex/logs/ik_right_seed7.json
```

1920x1080 녹화 예시:

```bash
PYTHONPATH=. python finger_ik_experiment.py --record --record-width 1920 --record-height 1080 --record-fps 60 --save-json codex/logs/ik_right_seed7.json
```
