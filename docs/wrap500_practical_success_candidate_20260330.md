# Wrap500 Practical-Success Candidate

## TL;DR
- Current candidate model:
  - `models/mustard_octo_official_wrap500_mix_k4_temporalcontinue_20260325/run_260325_161632/best_rollout_model`
- This is **not** a strict benchmark success model yet.
- This **is** a strong **practical-success candidate** because it can already:
  - reach the bottle,
  - form full wrap contact with thumb,
  - and lift above the success height threshold,
  - but still fails the current strict hold-continuity metric.

## Important Safety Note
- Do **not** describe this as "benchmark success achieved".
- The honest wording is:
  - "official Octo VLA rollout produced a practical near-success / practical-success candidate"
  - "strict wrap-hold benchmark success is still 0"
- This is not a fake scripted replay or oracle rollout.
- It is still the learned Octo policy running closed-loop in MuJoCo.

## Is This Really Octo / VLA Inference?
Yes.

The deployment path is still Octo-based VLA inference:
- model loading:
  - `scripts/eval/rollout_mustard_intent_octo_official.py:34`
- official Octo Gym wrappers:
  - `scripts/eval/rollout_mustard_intent_octo_official.py:35`
  - `scripts/eval/rollout_mustard_intent_octo_official.py:278`
  - `scripts/eval/rollout_mustard_intent_octo_official.py:282`
- goal/language task creation:
  - `scripts/eval/rollout_mustard_intent_octo_official.py:155`
  - `scripts/eval/rollout_mustard_intent_octo_official.py:168`
- learned action sampling from the Octo model:
  - `scripts/eval/rollout_mustard_intent_octo_official.py:314`

So the near-success clip is not coming from:
- scripted expert rollout,
- oracle replay,
- hand-authored action sequence,
- or a fake success override.

It is the learned Octo model sampling actions online from:
- current observation image,
- proprio/state,
- language instruction,
- goal image.

## Why It Is Still Not Strictly Successful
The strict evaluator requires more than "grasped and lifted once."

Strict wrap success currently requires:
- reaching the object:
  - `scripts/eval/mustard_intent_gym_env.py:121`
  - `scripts/eval/mustard_intent_gym_env.py:402`
- lifting above threshold:
  - `scripts/eval/mustard_intent_gym_env.py:122`
  - `scripts/eval/mustard_intent_gym_env.py:337`
- contact quality meeting the wrap rule continuously:
  - `scripts/data/collect_pickandlift_rlds.py:636`
  - `scripts/eval/mustard_intent_gym_env.py:338`
- and maintaining that condition for the full hold window:
  - `scripts/eval/mustard_intent_gym_env.py:339`
  - `scripts/eval/mustard_intent_gym_env.py:340`

The hold window is:
- `lift_hold_seconds = 1.5`
- effective control rate `100 Hz`
- so `lift_hold_steps = 150`
  - `scripts/eval/mustard_intent_gym_env.py:123`
  - `scripts/eval/mustard_intent_gym_env.py:180`

This means the model can still be marked `success = false` even if it:
- reaches,
- wraps with all fingers,
- and lifts high enough,
if that state is not held stably for long enough.

## Why We Still Consider It a Practical-Success Candidate
The reproduced near-success rollout is:
- summary:
  - `codex/logs/json/wrap500_temporalcontinue_bestroll_nearsuccess_seed2_20260330.json`
- video:
  - `codex/logs/videos/near_success_0330/mustard_intent_octo_official_wrap_and_lift_260330_114217.mp4`

That rollout achieved:
- `reached = true`
- `approach_min_err ≈ 0.0191 m`
- `best_contacts = 10`
- `best_fingers = ['ff','mf','rf','th']`
- `object_dz_max ≈ 0.1676`
- but `wrap_hold_hits = 0`

This is why it is reasonable to call it a practical-success candidate:
- object front alignment is good,
- full wrap happens,
- lift height is clearly sufficient,
- failure is now mostly hold continuity, not fundamental reach or grasp failure.

## Practical Success Definition We Recommend Internally
For internal qualitative tracking, use:

`success_practical = reached and full_wrap_contact and lifted`

Where:
- `reached` means the current arm target was reached.
- `full_wrap_contact` means thumb + 3 fingers participated.
- `lifted` means `object_dz_max >= 0.08`.

For publications / formal benchmark numbers, keep using:
- the current strict success metric.

## Data Collection Used For This Candidate
The current wrap500 pipeline was collected in three buckets.

### 1. Clean Anchor Set
- path:
  - `dataset/mustard_intent_wrap500_clean100_zupfix_20260324`
- size:
  - `100` successful episodes in `100` attempts
- summary:
  - `dataset/mustard_intent_wrap500_clean100_zupfix_20260324/collection_summary.json`
- key settings:
  - no spawn jitter
  - no arm disturbance
  - `capture_hz = 5`
  - `control_hz = 100`

### 2. Bulk Randomized Set
- path:
  - `dataset/mustard_intent_wrap500_bulk250_zupfix_20260324`
- size:
  - `250` successful episodes in `271` attempts
- summary:
  - `dataset/mustard_intent_wrap500_bulk250_zupfix_20260324/collection_summary.json`
- key settings:
  - `spawn_jitter_xy = 0.02`
  - `spawn_yaw_jitter_deg = 10`
  - `settle_steps = 40`
  - `arm_disturb_xyz = (0.01, 0.01, 0.005)`
  - disturb phases:
    - `approach,preshape`

### 3. Close-Focused Set
- path:
  - `dataset/mustard_intent_wrap500_close150_zupfix_20260324`
- size:
  - `150` successful episodes in `157` attempts
- summary:
  - `dataset/mustard_intent_wrap500_close150_zupfix_20260324/collection_summary.json`
- key settings:
  - `spawn_jitter_xy = 0.02`
  - `spawn_yaw_jitter_deg = 10`
  - `settle_steps = 40`
  - `arm_disturb_xyz = (0.006, 0.006, 0.003)`
  - disturb phases:
    - `preshape,close`

## Collector Behavior Used For This Dataset
This 500-episode set was not collected with the old rough lift behavior.
It used the fixed collector path:
- break on grasp acquisition,
- approach early-exit,
- retract then upright then world-z lift,
- `thumb_o_wrap` hand trajectory.

Representative values visible in the collection summaries:
- `approach_early_exit_pos_err = 0.01`
- `approach_early_exit_rot_err_deg = 2.0`
- `approach_early_exit_stable_steps = 3`
- `lift_target_mode = latched_palm_retract_then_upright_z_up`
- `post_grasp_upright_steps = 40`
- `post_grasp_retract_steps = 25`
- `post_grasp_retract_height = 0.03`

## Dataset Conversion Pipeline
The 500 episodes were merged and converted as:

1. raw bundle:
   - `dataset/mustard_intent_wrap500_mix_bundle_20260324`
2. shared k4 hand basis:
   - `dataset/mustard_intent_wrap500_mix_k4_basis_20260324/mustard_intent_hand_pca_k4.npz`
3. converted arm+hand-latent raw dataset:
   - `dataset/mustard_intent_wrap500_mix_k4_20260324`
4. official RLDS/OXE export:
   - `dataset/mustard_intent_wrap500_mix_k4_20260324_oxe`

The canonical orchestration script was:
- `codex/run_wrap500_collect_and_train_20260324.sh`

## Training History

### Stage A. First 500-wrap official train
- run:
  - `models/mustard_octo_official_wrap500_mix_k4_20260324/run_260324_182938`
- config:
  - `pretrained_path = hf://rail-berkeley/octo-base-1.5`
  - `window_size = 4`
  - `batch_size = 8`
  - `num_steps = 5000`
- result:
  - `best_train_step = 1127`
  - `best_train_loss ≈ 0.6430`
  - `best_rollout_step = 2000`
  - `best_rollout_score ≈ 217.76`

### Stage B. Validation-based continuation
- run:
  - `models/mustard_octo_official_wrap500_mix_k4_continue_20260325/run_260325_151712`
- started from:
  - `models/mustard_octo_official_wrap500_mix_k4_20260324/run_260324_182938/best_train_model`
- result:
  - `best_val_step = 4250`
  - `best_val_loss ≈ 0.7844`
  - `best_rollout_step = 1000`
  - `best_rollout_score ≈ 428.65`

### Stage C. Temporal-aligned continuation
- run:
  - `models/mustard_octo_official_wrap500_mix_k4_temporalcontinue_20260325/run_260325_161632`
- started from:
  - `models/mustard_octo_official_wrap500_mix_k4_continue_20260325/run_260325_151712/best_val_model`
- rollout selection changed to:
  - `wrapper = temporal`
  - `episodes = 3`
  - `spawn_jitter_xy = 0.02`
  - `spawn_yaw_jitter_deg = 10`
- result:
  - `best_train_step = 2103`
  - `best_train_loss ≈ 0.3384`
  - `best_val_step = 1750`
  - `best_val_loss ≈ 0.7574`
  - `best_rollout_step = 2000`
  - `best_rollout_score ≈ 1027.13`

This stage produced the current practical-success candidate.

## Why We Trust This Is Not a Fake Success
We are **not** claiming strict benchmark success.
That is the key honesty safeguard.

What we are claiming is narrower:
- the learned Octo VLA policy can sometimes produce:
  - correct reach,
  - full wrap contact,
  - and meaningful lift.

This claim is supported by:
- official Octo inference path,
- closed-loop MuJoCo rollout,
- saved video,
- saved JSON metrics,
- no oracle/scripted action injection during inference.

What remains unproven:
- stable success under the current strict hold metric,
- robust multi-seed success rate,
- real-robot transfer.

## Recommended Naming Going Forward
Use this phrasing:

- `strict_success_model`: no
- `practical_success_candidate`: yes
- `official Octo VLA model`: yes
- `scripted/oracle success`: no

This keeps us honest while still acknowledging that the behavior has improved to a meaningful level.
