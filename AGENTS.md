# AGENTS.md

This file stores durable collaboration rules for this repository.
When user-assistant decisions affect workflow or project scope, update this file.
Keep this rule itself in the file.

## Working Rules

1. Use `conda` environment `handvla` as the default runtime for install and checks.
2. Do not recreate `.venv` unless explicitly requested.
3. Put temporary experiment code, debug scripts, and logs in `codex/`.
4. Before final upload/push, clean up temporary files in `codex/` and keep only what is intentionally versioned.
5. Keep this file updated continuously with important decisions from user-assistant conversation.
6. Record this policy itself as durable context whenever AGENTS.md is updated.

## Current Project Baseline (2026-02-19)

1. Repository goal is to build from previous `keti_dual_arm_VLA` assets into a clean `handvla` project.
2. First runnable target is MuJoCo Allegro hand-only simulation.
3. `run_allegro_hand.py` launches the hand model with right control panel sliders.
4. Hand-only scene should include visible background/floor/lighting by default.
5. Default initial pose should be wrist-lower, hand-up orientation.
6. Background/lighting style should stay aligned with `keti_dual_arm_VLA` base scene unless user asks otherwise.
7. TCP markers should be defined near each fingertip pad, shifted slightly palm-ward for grasp experiments.
8. Viewer should start with a close frontal hand camera, not a distant wide shot.
9. Finger IK feasibility should be tested with per-finger random reachable targets and reported as success rate/error statistics.
10. Temporary logs stay under `codex/logs/`, but core experiment scripts should live in project root or stable modules.
11. `run_allegro_hand.py` should support 1920x1080 recording to `codex/logs/` by default when requested.
12. `finger_ik_experiment.py` should show viewer by default unless `--no-viewer` is passed.
13. IK experiment viewer should use the same default camera as `run_allegro_hand.py` and show target TCP markers clearly.
14. Default visible IK update speed should be around 20Hz (`--viewer-step`), with optional manual delay override.
15. Finger IK quick runs should default to 3 trials per finger.
16. Only active finger target marker should be visible during IK trials to avoid visual confusion.
17. `finger_ik_experiment.py` should support 1920x1080 recording options similar to `run_allegro_hand.py`.
