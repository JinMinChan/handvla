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
14. Default visible update speed for viewer-driven experiment scripts should be around 60Hz (`--viewer-step`), with optional manual delay override.
15. Finger IK quick runs should default to 3 trials per finger.
16. Only active finger target marker should be visible during IK trials to avoid visual confusion.
17. `finger_ik_experiment.py` should support 1920x1080 recording options similar to `run_allegro_hand.py`.
18. IK feasibility code and mustard grasp/data-collection code must stay in separate scripts.
19. `finger_ik_experiment.py` is IK-only (no mustard/object mode).
20. Mustard grasp pretraining scope is mustard-only (`006_mustard_bottle`) in `collect_mustard_grasp.py`.
21. Mustard fixed-air spawn pose uses palm-relative offset `[0.10, 0.00, 0.02]` with horizontal+clockwise quaternion `[0.5, 0.5, 0.5, -0.5]`.
22. Mustard object fixation uses freejoint pose clamp (kinematic lock) each simulation step.
23. Grasp success labeling uses geom contact between Allegro fingertip/distal geoms and mustard geoms with thumb-contact requirement plus contact/finger/force/stability thresholds.
24. Mustard grasp closing should use a thumb-opposition pre-shape phase (`open -> pre-shape -> close`) so the thumb rotates inward before full flexion.
25. Dataset collection for training should use `collect_mustard_grasp_dataset.py` and save only successful episodes until target count is reached.
26. Default mustard dataset logging rates are control 100Hz and capture 20Hz (via `capture_every=5`).
27. Mustard raw dataset path is `dataset/mustard_grasp/raw/episode_*.npz` with summary at `dataset/mustard_grasp/collection_summary.json`.
28. OXE conversion should use `convert_mustard_raw_to_oxe.py` to export RLDS/TFDS from raw episodes.
29. Octo fine-tuning should use `finetune_mustard_octo.py` with `mustard_grasp_oxe` (`image_primary` + `state` -> 12-DoF TCP action head: 4 fingertips x xyz).
30. Octo training runtime currently requires `conda` env `octoketi` (while `handvla` remains default for simulation/data tools).
31. Fine-tuning outputs should be stored under `models/mustard_octo_overfit/run_*` with checkpoints, `final_model`, and loss trace.
32. `finetune_mustard_octo.py` should use direct TFDS episode loading (not `make_single_dataset`) for compatibility with current `octoketi` TFDS stack.
33. Current Octo fine-tune script supports `window_size=1` (single-frame policy training).
34. Octo loss compatibility requires `action` and `action_pad_mask` batch tensors shaped as `[B, window(1), action_horizon, action_dim]`.
35. Finetuned model rollout validation should use `rollout_mustard_octo.py` in the same fixed mustard scene with contact-based success metrics.
36. Default stable rollout settings for current overfit model are `max_policy_steps=100`, `control_repeat=5`, `action_smoothing=0.35`.
37. `rollout_mustard_octo.py` should run in `conda` env `handvla` after Octo-compatible dependencies are installed.
38. `rollout_mustard_octo.py` auto-resolves Octo source path from `/home/minchan/Downloads/dual_arm_VLA/octo`, or from project-root `.octo_src_path` override.
39. `collect_mustard_grasp_dataset.py` default `--action-interface` is `tcp12`; existing legacy datasets may still contain `joint16`.
40. `rollout_mustard_octo.py` default `--action-interface` is `tcp12`; use `--action-interface joint16` only for legacy 16-DoF models.
41. Legacy 16-DoF artifacts should use `full_joint` naming (`dataset/mustard_grasp_full_joint`, `dataset/mustard_grasp_oxe_full_joint`, `models/mustard_octo_overfit_full_joint`).
42. Keep presentation-ready comparison videos under `codex/logs/` with explicit interface in filename (e.g., `full_joint_*.mp4`).
43. Rollout recording should keep the default frontal hand camera even when running with `--no-viewer` (headless).
44. Use `rollout_mustard_octo_tcp12.py` for tcp12-specific rollout debugging (step-level `tcp_delta`/`q_delta` logging).
45. TCP12 dataset semantics are command next-delta: `action_t = FK(q_cmd_{t+1}) - FK(q_cmd_t)` (final step repeats previous delta); default frame is `palm_local` (legacy option: `world`).
46. TCP12 rollout interpretation default is delta mode (`tcp_target = tcp_now + pred_delta_world`), where `palm_local` predictions are rotated to world using current palm orientation; absolute mode remains legacy compatibility only.
47. Oracle replay diagnostic (2026-02-26): `joint16` replay reproduces grasp success (100/100), while `tcp12` (`delta` and `absolute`) replay gives 0/100 under thumb-required success criteria.
48. Under current geom-contact success rules, `tcp12` position-only replay consistently misses thumb contact (`best_fingers` tends to `ff,mf,rf`), indicating interface loss (thumb posture/opposition) rather than pure model-learning failure.
49. Validation note: if thumb-contact requirement is disabled, oracle replay success for `tcp12` becomes 100/100; therefore future 12-DoF interface work must encode thumb opposition intent/posture (or revise success/target definition) before fair comparison with full-joint control.
50. Policy-direction diagnostic (2026-02-26): current tcp12-delta model outputs for `ff/mf/rf` are weaker than demonstration scale and often point away from object center, while thumb output dominates; this matches observed thumb-only contact in rollout videos.
51. Keep `codex/logs/tcp12_direction_diagnostics_20260226.json` as reference when discussing why tcp12 VLA behavior differs from per-finger IK feasibility tests.
52. Next primary low-dimensional interface direction is `joint-synergy kD` (joint-space compression) rather than naive `tcp12 xyz-only`; detailed execution plan is tracked in `docs/joint-synergy-kD.md`.
53. Maintain a separate presentation-oriented experiment narrative document at `docs/experiment-flow-for-presentation.md` so slide preparation can follow the verified experimental timeline and conclusions.
54. Joint-synergy tooling is implemented with `build_joint_synergy_basis.py`, `convert_full_joint_raw_to_synergy_raw.py`, and `rollout_mustard_octo_synergy.py`.
55. Current full_joint mustard dataset is effectively low-rank; PCA basis selection run (k=4,6,8,10) produced oracle success 100% for all tested k and selected `k=4` as the smallest valid default (`dataset/synergy_basis/full_joint_pca_k4.npz`).
56. Current synergy experiment artifacts: raw `dataset/mustard_grasp_synergy_k4`, OXE `dataset/mustard_grasp_oxe_synergy_k4`, model `models/mustard_octo_overfit_synergy_k4/run_260227_115921`, eval `codex/logs/synergy_k4_eval_summary_20260227.json`.
57. Initial synergy rollout check with the above `k=4` model achieved multi-finger + thumb contact and 100% success on a 3-episode smoke evaluation.
58. `collect_mustard_grasp_dataset.py` now supports optional diversity controls (spawn jitter, yaw jitter, pose noise, phase-step jitter) and should keep default-zero behavior for legacy deterministic collection.
59. Diverse benchmark dataset uses `dataset/mustard_grasp_full_joint_diverse` with diversity config: xy 0.02m, z 0.01m, yaw 20deg, preshape noise 0.05rad, close noise 0.08rad, step jitter (4,5,15,8), seed 42.
60. Diverse synergy basis run (`dataset/synergy_basis_diverse/full_joint_pca_summary.json`) selected `k=4` with cumulative variance ~0.992 and oracle success 1.0.
61. Diverse fair-training artifacts: full-joint model `models/mustard_octo_overfit_full_joint_diverse/run_260228_011500`, synergy model `models/mustard_octo_overfit_synergy_k4_diverse/run_260228_012422`.
62. Diverse 10-episode rollout evaluation reached 100% success for both interfaces; synergy-k4 showed higher best-contact-finger count (4.0 mean) than full-joint baseline (3.0 mean) in current fixed-scene benchmark.
63. Presentation-ready diverse comparison outputs are `codex/logs/diverse_joint_synergy_comparison_20260228.json`, `codex/logs/diverse_joint_synergy_comparison_20260228.md`, plus videos `codex/logs/full_joint_diverse_rollout_demo.mp4` and `codex/logs/synergy_k4_diverse_rollout_demo.mp4`.
64. `rollout_mustard_octo.py` supports `--show-fk-tcp-markers` to visualize FK TCP world positions as blue mocap spheres for full-joint-to-TCP extraction validation and recording.
65. `collect_mustard_grasp_dataset.py` stores auxiliary command traces in raw episodes: `action_joint16_cmd`, `action_tcp12_cmd_world_abs`, `action_tcp12_cmd_world_next_delta`, and `action_tcp12_cmd_palm_local_next_delta` for replay/debug parity checks.
66. Current tcp12 interface-fix stage keeps action dimension at 12 (no extra intent dims yet); focus is command semantics/frame consistency first.
67. Historical command-based tcp12 artifacts (obs->cmd delta stage, 2026-03-03): raw `dataset/mustard_grasp_tcp12_cmd`, OXE `dataset/mustard_grasp_oxe_tcp12_cmd`, model `models/mustard_octo_overfit_tcp12_cmd/run_260303_153616`.
68. TCP12 semantics comparison eval (2026-03-03, 10 episodes each): legacy-world and command-palm_local (obs->cmd delta stage) both scored 0% success under thumb-required rule; command model improved contact occurrence but remained thumb-only. See `codex/logs/tcp12_legacy_world_eval_summary_20260303.json`, `codex/logs/tcp12_cmd_palm_local_eval_summary_20260303.json`, and `codex/logs/tcp12_semantics_comparison_20260303.{json,md}` with demo videos `codex/logs/tcp12_legacy_world_rollout_demo_20260303.mp4`, `codex/logs/tcp12_cmd_palm_local_rollout_demo_20260303.mp4`.
69. Current cmd-next tcp12 artifacts (2026-03-03): raw `dataset/mustard_grasp_tcp12_cmd_next`, OXE `dataset/mustard_grasp_oxe_tcp12_cmd_next`, model `models/mustard_octo_overfit_tcp12_cmd_next/run_260303_171027`.
70. Cmd-next tcp12 10-episode rollout (2026-03-03, thumb-required) with `models/mustard_octo_overfit_tcp12_cmd_next/run_260303_171027/final_model` still produced 0% success and 0 contact-fingers across episodes; step logs show early motion followed by near-zero updates (`q_delta -> 0`). Demo video: `codex/logs/tcp12_cmd_next_palm_local_rollout_demo_20260303.mp4`.
71. Expanded diverse synergy basis sweep (2026-03-04) tested `k=2,3,4,6` at `dataset/synergy_basis_diverse_k236/full_joint_pca_summary.json`; all ks achieved oracle replay success 1.0, and `k=2` already retained high variance (~0.9899 cumulative).
72. Diverse k-ablation artifacts (2026-03-04): raw `dataset/mustard_grasp_synergy_k{2,3,4,6}_diverse_k236`, OXE `dataset/mustard_grasp_oxe_synergy_k{2,3,4,6}_diverse_k236`, models `models/mustard_octo_overfit_synergy_k{2,3,4,6}_diverse_k236/run_260304_*`, eval summaries `codex/logs/synergy_k{2,3,4,6}_diverse_k236_eval_20260304.json`.
73. Diverse 5000-step rollout eval for `k=2,3,4,6` (10 episodes each, thumb-required) all reached 100% success with 4 contact fingers; comparison table is tracked in `codex/logs/synergy_k_ablation_20260304.{json,md}`.
74. User-selected follow-up baseline (2026-03-04) is `k=4`; dedicated retrain/eval artifacts are `models/mustard_octo_overfit_synergy_k4_selected/run_260304_140028`, `codex/logs/synergy_k4_selected_eval_20260304.json`, and `codex/logs/synergy_k4_selected_summary_20260304.{json,md}`.
75. Paper direction is fixed to `action-interface co-design` for unified arm-hand VLA (not just end-to-end demonstration); ICCAS draft execution plan is tracked in `docs/iccas-action-interface-plan.md`.
76. Before upload, keep only intentionally versioned outputs and prioritize pruning heavy experiment byproducts under `dataset/`, `models/`, and temporary `codex/tmp_*` directories.
77. A repeatable pre-push cleanup script is available at `tools/cleanup_before_push.sh`; default keep set preserves only current core comparison artifacts (`full_joint_diverse` vs `synergy_k4_selected`) and synergy basis files.
78. `.gitignore` now ignores `dataset/` and `models/` by default to prevent accidental large artifact uploads; keep release assets separately if explicitly needed.
79. Log/media pruning policy before upload: keep one representative rollout video per algorithm family in `codex/logs/` (`full_joint`, `tcp12`, `synergy-k4`, plus optional IK feasibility), and remove redundant benchmark/debug clips.
80. Canonical executable entrypoints are now organized under `scripts/` by role: `scripts/sim`, `scripts/data`, `scripts/train`, `scripts/eval`, `scripts/research`.
81. Research/design documents are organized under `docs/` (`docs/iccas-action-interface-plan.md`, `docs/joint-synergy-kD.md`, `docs/experiment-flow-for-presentation.md`).
82. Log/dataset/model cleanup helpers are organized under `tools/` (`tools/cleanup_before_push.sh`, `tools/prune_logs_representative.sh`) and should be run from repository root (scripts auto-resolve root).
