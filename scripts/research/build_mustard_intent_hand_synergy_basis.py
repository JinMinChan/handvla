#!/usr/bin/env python3
"""Build and evaluate hand-only PCA bases for the mustard intent benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from datetime import datetime
import json
import shutil
from types import SimpleNamespace

import mujoco
import numpy as np

from env import franka_allegro_mjcf
from scripts.data.collect_mustard_intent_benchmark import (
    PHASE_PULL,
    PHASE_PUSH,
    TASK_HOOK_AND_PULL,
    TASK_PUSH_OVER,
    TASK_WRAP_AND_LIFT,
    _contact_meets,
    _pull_contact_meets,
    _push_contact_meets,
)
from scripts.data.collect_pickandlift_rlds import (
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_LIFT,
    PHASE_LIFT_HOLD,
    PHASE_PRESHAPE,
    PHASE_SETTLE,
    _build_arm_config,
    _build_contact_config,
    _build_hand_config,
    _build_mustard_config,
    _detect_contact_with_target,
    _quat_lerp_normalize,
    _quat_to_rot,
    _rot_to_quat,
    _set_mustard_pose,
)


LOCKED_PHASES = {PHASE_SETTLE, PHASE_APPROACH, PHASE_PRESHAPE}
SUPPORTED_TASKS = (TASK_WRAP_AND_LIFT, TASK_PUSH_OVER, TASK_HOOK_AND_PULL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit PCA hand bases on the mustard intent benchmark and choose the best k "
            "using oracle replay with original arm actions plus reconstructed hand actions."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset/mustard_intent_v1_common_pose_20260311",
        help="Root directory containing per-task raw episode folders.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="wrap_and_lift,push_over,hook_and_pull",
        help="Comma-separated task names to include.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/mustard_intent_synergy_basis_20260311",
        help="Directory for basis files and summary json.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="2,3,4,5,6,7,8",
        help="Comma-separated latent dimensions to evaluate.",
    )
    parser.add_argument(
        "--hand-action-start",
        type=int,
        default=7,
        help="Start index of hand joints in the raw action tensor.",
    )
    parser.add_argument(
        "--hand-action-dim",
        type=int,
        default=16,
        help="Number of hand joint action dims in the raw action tensor.",
    )
    parser.add_argument(
        "--save-best-alias",
        action="store_true",
        default=True,
        help="Also save mustard_intent_hand_pca_best.npz alias for the selected k.",
    )
    parser.add_argument(
        "--no-save-best-alias",
        dest="save_best_alias",
        action="store_false",
        help="Disable best-alias export.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fit and summarize without writing basis files.",
    )
    parser.add_argument(
        "--post-settle-steps",
        type=int,
        default=500,
        help=(
            "Extra raw MuJoCo steps to simulate after the recorded action sequence for "
            "dynamic tasks like push/pull so delayed object motion is counted."
        ),
    )
    return parser.parse_args()


def _parse_csv(text: str) -> list[str]:
    out = [tok.strip() for tok in text.split(",") if tok.strip()]
    if not out:
        raise ValueError("No values parsed.")
    return out


def _parse_k_values(text: str) -> list[int]:
    values = sorted({int(tok) for tok in _parse_csv(text)})
    if any(k <= 0 for k in values):
        raise ValueError("k must be >= 1.")
    return values


def _string_scalar(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def _json_scalar(value) -> dict:
    text = _string_scalar(value)
    return json.loads(text)


def _fit_pca(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    Xc = X - mu
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    dof = max(int(X.shape[0]) - 1, 1)
    explained_var = (s**2) / dof
    explained_ratio = explained_var / max(float(np.sum(explained_var)), 1e-12)
    cum_ratio = np.cumsum(explained_ratio)
    return mu, vt, explained_ratio, cum_ratio


def _rot_err_deg(current_quat_wxyz: np.ndarray, target_quat_wxyz: np.ndarray) -> float:
    r_cur = _quat_to_rot(np.asarray(current_quat_wxyz, dtype=np.float64))
    r_tgt = _quat_to_rot(np.asarray(target_quat_wxyz, dtype=np.float64))
    r_delta = r_cur.T @ r_tgt
    trace = float(np.trace(r_delta))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_angle)))


def _tilt_deg(rot: np.ndarray) -> float:
    z_axis = np.asarray(rot, dtype=np.float64).reshape(3, 3)[:, 2]
    return float(np.rad2deg(np.arccos(np.clip(z_axis[2], -1.0, 1.0))))


def _load_episodes(
    dataset_root: Path,
    tasks: list[str],
    hand_start: int,
    hand_dim: int,
) -> tuple[list[dict], np.ndarray]:
    hand_end = hand_start + hand_dim
    episodes: list[dict] = []
    flat_hand: list[np.ndarray] = []
    side_set: set[str] = set()

    for task in tasks:
        raw_dir = dataset_root / task / "raw"
        files = sorted(p for p in raw_dir.glob("episode_*.npz") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No episode files found under: {raw_dir}")

        for path in files:
            data = np.load(path, allow_pickle=True)
            action = np.asarray(data["action"], dtype=np.float64)
            if action.ndim != 2 or action.shape[1] < hand_end:
                raise ValueError(f"Unexpected action shape {action.shape} in {path}")
            flat_hand.append(action[:, hand_start:hand_end])
            side = _string_scalar(data["side"])
            side_set.add(side)
            episodes.append(
                {
                    "path": path,
                    "task_name": _string_scalar(data["task_name"]),
                    "action": action,
                    "phase": np.asarray(data["phase"], dtype=np.int32).reshape(-1),
                    "object_qpos": np.asarray(data["object_qpos"], dtype=np.float64).reshape(7),
                    "arm_cmd_pose_wxyz": np.asarray(
                        data["arm_cmd_pose_wxyz"], dtype=np.float64
                    ).reshape(-1, 7),
                    "criteria": _json_scalar(data["criteria_json"]),
                    "control_hz": float(np.asarray(data["control_hz"]).reshape(())),
                    "capture_hz": float(np.asarray(data["capture_hz"]).reshape(())),
                    "success": bool(np.asarray(data["success"]).reshape(())),
                    "side": side,
                }
            )

    if len(side_set) != 1:
        raise ValueError(f"Mixed sides in dataset: {sorted(side_set)}")
    X = np.concatenate(flat_hand, axis=0)
    if X.shape[1] != hand_dim:
        raise ValueError(f"Expected hand_dim={hand_dim}, got {X.shape[1]}")
    return episodes, X


def _build_env(side: str):
    mjcf = franka_allegro_mjcf.load(side=side, add_mustard=True, add_frame_axes=False)
    model = mjcf.compile()
    data = mujoco.MjData(model)
    arm_cfg = _build_arm_config(model, side)
    hand_cfg = _build_hand_config(model, side)
    mustard_cfg = _build_mustard_config(model)
    contact_cfg = _build_contact_config(model, side, mustard_cfg.body_id)
    return model, data, arm_cfg, hand_cfg, mustard_cfg, contact_cfg


def _choose_best_k(metrics_by_k: dict[int, dict], k_values: list[int]) -> int:
    ranked = sorted(
        k_values,
        key=lambda k: (
            float(metrics_by_k[k]["oracle_success_rate"]),
            float(metrics_by_k[k]["oracle_task_success_rate_min"]),
            -int(k),
        ),
        reverse=True,
    )
    return int(ranked[0])


def _oracle_eval_episode(
    *,
    model,
    data,
    arm_cfg,
    hand_cfg,
    mustard_cfg,
    contact_cfg,
    episode: dict,
    mu: np.ndarray,
    B: np.ndarray,
    hand_start: int,
    hand_dim: int,
    post_settle_steps: int,
) -> dict:
    initial_state = model.key("initial_state").id
    mujoco.mj_resetDataKeyframe(model, data, initial_state)
    mujoco.mj_forward(model, data)

    force_buf = np.zeros(6, dtype=float)
    spawn_pos = episode["object_qpos"][:3].astype(np.float64)
    spawn_quat = episode["object_qpos"][3:7].astype(np.float64)
    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
    mujoco.mj_forward(model, data)

    action = episode["action"]
    phases = episode["phase"]
    arm_cmd_pose = episode["arm_cmd_pose_wxyz"]
    criteria = SimpleNamespace(**episode["criteria"])
    control_repeat = max(1, int(round(episode["control_hz"] / max(episode["capture_hz"], 1e-6))))

    hand_end = hand_start + hand_dim
    hand_gt = action[:, hand_start:hand_end]
    arm_gt = action[:, :hand_start]
    Z = (hand_gt - mu[None, :]) @ B
    hand_recon = Z @ B.T + mu[None, :]
    hand_recon = np.clip(hand_recon, hand_cfg.q_min[None, :], hand_cfg.q_max[None, :])

    step_counter = 0
    best_contacts = 0
    best_force = 0.0
    best_fingers: set[str] = set()
    approach_min_err = np.inf
    approach_min_rot_err = np.inf
    grasp_stable_hits = 0
    grasp_acquired = False
    first_grasp_step = -1
    first_success_step = -1

    object_z_ref = float(spawn_pos[2])
    object_z_max = object_z_ref
    object_xy_ref = spawn_pos[:2].copy()
    initial_obj_yaw = float(
        np.arctan2(
            _quat_to_rot(spawn_quat)[1, 0],
            _quat_to_rot(spawn_quat)[0, 0],
        )
    )
    initial_x = float(spawn_pos[0])

    task_name = episode["task_name"]
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported oracle task: {task_name}")

    task_success = False
    no_drop_during_hold = True
    hold_contact_ok = True
    saw_hold = False
    topple_angle_max = 0.0
    push_contact_seen = False
    release_after_topple = False
    release_after_topple_hits = 0
    push_planar_disp_max = 0.0
    pull_dx_max = 0.0
    hook_contact_seen = False

    def _update_metrics(phase: int, cmd_pose: np.ndarray) -> None:
        nonlocal approach_min_err
        nonlocal approach_min_rot_err
        nonlocal best_contacts
        nonlocal best_force
        nonlocal best_fingers
        nonlocal grasp_stable_hits
        nonlocal grasp_acquired
        nonlocal first_grasp_step
        nonlocal task_success
        nonlocal first_success_step
        nonlocal no_drop_during_hold
        nonlocal hold_contact_ok
        nonlocal saw_hold
        nonlocal topple_angle_max
        nonlocal push_contact_seen
        nonlocal release_after_topple
        nonlocal release_after_topple_hits
        nonlocal push_planar_disp_max
        nonlocal object_z_max
        nonlocal pull_dx_max
        nonlocal hook_contact_seen
        nonlocal step_counter

        if phase in LOCKED_PHASES:
            _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
            mujoco.mj_forward(model, data)

        ee_pos_pre = data.xpos[arm_cfg.palm_body_id].copy()
        ee_quat_pre = _rot_to_quat(data.xmat[arm_cfg.palm_body_id].reshape(3, 3))
        if phase == PHASE_APPROACH:
            approach_min_err = min(approach_min_err, float(np.linalg.norm(cmd_pose[:3] - ee_pos_pre)))
            approach_min_rot_err = min(
                approach_min_rot_err,
                _rot_err_deg(ee_quat_pre, cmd_pose[3:7]),
            )

        n_contacts, total_force, touched = _detect_contact_with_target(
            model, data, contact_cfg, force_buf
        )
        object_pos = data.xpos[mustard_cfg.body_id].copy()
        object_rot = data.xmat[mustard_cfg.body_id].reshape(3, 3).copy()
        object_z = float(object_pos[2])
        object_z_max = max(object_z_max, object_z)
        push_planar_disp = float(np.linalg.norm(object_pos[:2] - object_xy_ref))
        push_planar_disp_max = max(push_planar_disp_max, push_planar_disp)

        best_contacts = max(best_contacts, int(n_contacts))
        best_force = max(best_force, float(total_force))
        if len(touched) >= len(best_fingers):
            best_fingers = set(touched)

        if task_name == TASK_WRAP_AND_LIFT:
            if phase == PHASE_CLOSE:
                meets = _contact_meets(n_contacts, total_force, touched, criteria)
                grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                if grasp_stable_hits >= criteria.stable_steps and not grasp_acquired:
                    grasp_acquired = True
                    first_grasp_step = step_counter
            elif phase in (PHASE_LIFT, PHASE_LIFT_HOLD) and grasp_acquired:
                lifted = (object_z - object_z_ref) >= criteria.lift_success_delta
                meets = _contact_meets(n_contacts, total_force, touched, criteria)
                if phase == PHASE_LIFT_HOLD:
                    saw_hold = True
                    if not lifted:
                        no_drop_during_hold = False
                    if not meets:
                        hold_contact_ok = False
            # Match the collector's success semantics exactly. Episodes were stored as
            # successful if the object stayed lifted through the hold phase; collector
            # did not require contact to remain above threshold for every hold step.
            task_success = bool(grasp_acquired and saw_hold and no_drop_during_hold)

        elif task_name == TASK_PUSH_OVER:
            if phase == PHASE_PUSH:
                topple_angle_max = max(topple_angle_max, _tilt_deg(object_rot))
                push_contact_seen = push_contact_seen or _push_contact_meets(
                    n_contacts, total_force, touched, criteria
                )
                toppled = topple_angle_max >= criteria.push_success_tilt_deg
                not_lifted = (object_z_max - object_z_ref) <= criteria.push_max_lift_dz
                released = int(n_contacts) == 0
                if toppled and not_lifted:
                    release_after_topple_hits = release_after_topple_hits + 1 if released else 0
                    release_after_topple = (
                        release_after_topple
                        or release_after_topple_hits
                        >= max(int(getattr(criteria, "push_release_steps", 3)), 1)
                    )
                else:
                    release_after_topple_hits = 0
            task_success = bool(
                topple_angle_max >= criteria.push_success_tilt_deg
                and (object_z_max - object_z_ref) <= criteria.push_max_lift_dz
                and release_after_topple
            )

        elif task_name == TASK_HOOK_AND_PULL:
            if phase in (PHASE_CLOSE, PHASE_PULL):
                hook_contact_seen = hook_contact_seen or _pull_contact_meets(
                    n_contacts, total_force, touched, criteria
                )
            if phase == PHASE_PULL:
                pull_dx = max(0.0, initial_x - float(object_pos[0]))
                pull_dx_max = max(pull_dx_max, pull_dx)
            task_success = bool(
                hook_contact_seen
                and pull_dx_max >= criteria.pull_success_dx
                and (object_z_max - object_z_ref) <= criteria.pull_max_lift_dz
            )

        if task_success and first_success_step < 0:
            first_success_step = step_counter

        step_counter += 1

    def _step_once(arm_cmd: np.ndarray, hand_cmd: np.ndarray, phase: int, cmd_pose: np.ndarray) -> None:
        data.ctrl[arm_cfg.act_ids] = arm_cmd.astype(np.float32)
        data.ctrl[7:23] = hand_cmd.astype(np.float32)
        if phase in LOCKED_PHASES:
            _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
        mujoco.mj_step(model, data, nstep=1)
        _update_metrics(phase, cmd_pose)

    # Replay sampled actions with linear interpolation inside each capture interval.
    # The collector generated commands at raw control frequency and only stored every
    # `capture_every`-th step, so zero-order hold over the sparse samples is not faithful.
    num_samples = action.shape[0]
    for t in range(num_samples):
        phase = int(phases[t])
        arm_cmd_t = arm_gt[t].astype(np.float64)
        hand_cmd_t = hand_recon[t].astype(np.float64)
        cmd_pose_t = arm_cmd_pose[t].astype(np.float64)

        if t < num_samples - 1:
            arm_cmd_next = arm_gt[t + 1].astype(np.float64)
            hand_cmd_next = hand_recon[t + 1].astype(np.float64)
            cmd_pose_next = arm_cmd_pose[t + 1].astype(np.float64)
            for sub in range(control_repeat):
                alpha = float(sub) / float(max(control_repeat, 1))
                arm_cmd = ((1.0 - alpha) * arm_cmd_t + alpha * arm_cmd_next).astype(np.float64)
                hand_cmd = ((1.0 - alpha) * hand_cmd_t + alpha * hand_cmd_next).astype(np.float64)
                cmd_pos = ((1.0 - alpha) * cmd_pose_t[:3] + alpha * cmd_pose_next[:3]).astype(np.float64)
                cmd_quat = _quat_lerp_normalize(cmd_pose_t[3:7], cmd_pose_next[3:7], alpha).astype(
                    np.float64
                )
                cmd_pose = np.concatenate([cmd_pos, cmd_quat], axis=0)
                _step_once(arm_cmd, hand_cmd, phase, cmd_pose)
        else:
            for _ in range(control_repeat):
                _step_once(arm_cmd_t, hand_cmd_t, phase, cmd_pose_t)

    if task_name in (TASK_PUSH_OVER, TASK_HOOK_AND_PULL) and post_settle_steps > 0:
        final_arm_cmd = action[-1, :hand_start].astype(np.float32)
        final_hand_cmd = np.clip(
            hand_recon[-1].astype(np.float32),
            hand_cfg.q_min.astype(np.float32),
            hand_cfg.q_max.astype(np.float32),
        )
        for _ in range(int(post_settle_steps)):
            data.ctrl[arm_cfg.act_ids] = final_arm_cmd
            data.ctrl[7:23] = final_hand_cmd
            mujoco.mj_step(model, data, nstep=1)

            n_contacts, total_force, touched = _detect_contact_with_target(
                model, data, contact_cfg, force_buf
            )
            object_pos = data.xpos[mustard_cfg.body_id].copy()
            object_rot = data.xmat[mustard_cfg.body_id].reshape(3, 3).copy()
            object_z = float(object_pos[2])
            object_z_max = max(object_z_max, object_z)
            best_contacts = max(best_contacts, int(n_contacts))
            best_force = max(best_force, float(total_force))
            if len(touched) >= len(best_fingers):
                best_fingers = set(touched)
            push_planar_disp = float(np.linalg.norm(object_pos[:2] - object_xy_ref))
            push_planar_disp_max = max(push_planar_disp_max, push_planar_disp)

            if task_name == TASK_PUSH_OVER:
                topple_angle_max = max(topple_angle_max, _tilt_deg(object_rot))
                toppled = topple_angle_max >= criteria.push_success_tilt_deg
                not_lifted = (object_z_max - object_z_ref) <= criteria.push_max_lift_dz
                released = int(n_contacts) == 0
                if toppled and not_lifted:
                    release_after_topple_hits = release_after_topple_hits + 1 if released else 0
                    release_after_topple = (
                        release_after_topple
                        or release_after_topple_hits
                        >= max(int(getattr(criteria, "push_release_steps", 3)), 1)
                    )
                else:
                    release_after_topple_hits = 0
            elif task_name == TASK_HOOK_AND_PULL:
                pull_dx = max(0.0, initial_x - float(object_pos[0]))
                pull_dx_max = max(pull_dx_max, pull_dx)

        # Dynamic tasks can satisfy success only after the commanded motion ends.
        # Re-evaluate task success using the post-settle extrema collected above.
        if task_name == TASK_PUSH_OVER:
            task_success = bool(
                topple_angle_max >= criteria.push_success_tilt_deg
                and (object_z_max - object_z_ref) <= criteria.push_max_lift_dz
                and release_after_topple
            )
        elif task_name == TASK_HOOK_AND_PULL:
            task_success = bool(
                hook_contact_seen
                and pull_dx_max >= criteria.pull_success_dx
                and (object_z_max - object_z_ref) <= criteria.pull_max_lift_dz
            )

        if task_success and first_success_step < 0:
            first_success_step = step_counter

    reached = bool(approach_min_err <= criteria.arm_reach_threshold)
    success = bool(reached and task_success)
    metrics = {
        "task_name": task_name,
        "reached": reached,
        "task_success": bool(task_success),
        "success": success,
        "approach_min_err": float(approach_min_err),
        "approach_min_rot_err_deg": float(approach_min_rot_err),
        "first_grasp_step": int(first_grasp_step),
        "first_success_step": int(first_success_step),
        "best_contacts": int(best_contacts),
        "best_force": float(best_force),
        "best_fingers": sorted(best_fingers),
    }
    if task_name == TASK_WRAP_AND_LIFT:
        metrics.update(
            {
                "object_dz_max": float(object_z_max - object_z_ref),
                "no_drop_during_hold": bool(no_drop_during_hold),
                "hold_contact_ok": bool(hold_contact_ok),
            }
        )
    elif task_name == TASK_PUSH_OVER:
        metrics.update(
            {
                "tilt_deg_max": float(topple_angle_max),
                "push_contact_seen": bool(push_contact_seen),
                "release_after_topple": bool(release_after_topple),
                "push_planar_disp_max": float(push_planar_disp_max),
                "object_xy_ref": [float(object_xy_ref[0]), float(object_xy_ref[1])],
                "object_xy_final": [float(object_pos[0]), float(object_pos[1])],
                "object_dz_max": float(object_z_max - object_z_ref),
            }
        )
    elif task_name == TASK_HOOK_AND_PULL:
        metrics.update(
            {
                "pull_dx_max": float(pull_dx_max),
                "hook_contact_seen": bool(hook_contact_seen),
                "object_dz_max": float(object_z_max - object_z_ref),
            }
        )
    return metrics


def _oracle_eval(
    episodes: list[dict],
    mu: np.ndarray,
    B: np.ndarray,
    hand_start: int,
    hand_dim: int,
    post_settle_steps: int,
) -> dict:
    model, data, arm_cfg, hand_cfg, mustard_cfg, contact_cfg = _build_env(episodes[0]["side"])
    task_stats = {
        task: {"episodes": 0, "successes": 0, "best_contacts": [], "best_fingers": []}
        for task in SUPPORTED_TASKS
    }
    detailed = []

    for episode in episodes:
        metrics = _oracle_eval_episode(
            model=model,
            data=data,
            arm_cfg=arm_cfg,
            hand_cfg=hand_cfg,
            mustard_cfg=mustard_cfg,
            contact_cfg=contact_cfg,
            episode=episode,
            mu=mu,
            B=B,
            hand_start=hand_start,
            hand_dim=hand_dim,
            post_settle_steps=post_settle_steps,
        )
        detailed.append({"episode_path": str(episode["path"]), **metrics})
        stats = task_stats[metrics["task_name"]]
        stats["episodes"] += 1
        stats["successes"] += int(metrics["success"])
        stats["best_contacts"].append(float(metrics["best_contacts"]))
        stats["best_fingers"].append(float(len(metrics["best_fingers"])))

    total_eps = sum(stats["episodes"] for stats in task_stats.values())
    total_success = sum(stats["successes"] for stats in task_stats.values())
    by_task = {}
    for task, stats in task_stats.items():
        episodes_n = max(stats["episodes"], 1)
        by_task[task] = {
            "episodes": int(stats["episodes"]),
            "success_rate": float(stats["successes"] / episodes_n) if stats["episodes"] else 0.0,
            "best_contacts_mean": float(np.mean(stats["best_contacts"])) if stats["best_contacts"] else 0.0,
            "best_fingers_mean": float(np.mean(stats["best_fingers"])) if stats["best_fingers"] else 0.0,
        }

    return {
        "episodes": int(total_eps),
        "success_rate": float(total_success / max(total_eps, 1)),
        "successes": int(total_success),
        "by_task": by_task,
        "detailed": detailed,
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tasks = _parse_csv(args.tasks)
    for task in tasks:
        if task not in SUPPORTED_TASKS:
            raise SystemExit(f"Unsupported task in --tasks: {task}")
    k_values = _parse_k_values(args.k_values)
    hand_start = int(args.hand_action_start)
    hand_dim = int(args.hand_action_dim)
    hand_end = hand_start + hand_dim

    episodes, X = _load_episodes(dataset_root, tasks, hand_start, hand_dim)
    mu, vt, explained_ratio, cum_ratio = _fit_pca(X)

    print(
        f"[intent-basis] dataset={dataset_root.name} episodes={len(episodes)} "
        f"transitions={X.shape[0]} action_dim=23 hand_slice=[{hand_start}:{hand_end}] "
        f"k_values={k_values}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_k: dict[int, dict] = {}
    basis_path_by_k: dict[int, Path] = {}

    for k in k_values:
        if k > vt.shape[0]:
            raise SystemExit(f"k={k} exceeds max rank {vt.shape[0]}")
        B = vt[:k].T.astype(np.float32)
        Z = (X - mu[None, :]) @ B
        Xhat = Z @ B.T + mu[None, :]
        err = Xhat - X
        rmse = float(np.sqrt(np.mean(np.square(err))))
        mean_abs = float(np.mean(np.abs(err)))
        max_abs = float(np.max(np.abs(err)))

        oracle = _oracle_eval(
            episodes,
            mu.astype(np.float32),
            B,
            hand_start,
            hand_dim,
            int(args.post_settle_steps),
        )
        task_rates = [float(oracle["by_task"][task]["success_rate"]) for task in tasks]

        basis_path = out_dir / f"mustard_intent_hand_pca_k{k}.npz"
        basis_path_by_k[k] = basis_path
        metrics = {
            "k": int(k),
            "explained_variance_ratio_k": float(np.sum(explained_ratio[:k])),
            "cumulative_explained_variance_ratio_k": float(cum_ratio[k - 1]),
            "reconstruction": {
                "rmse": rmse,
                "mean_abs_error": mean_abs,
                "max_abs_error": max_abs,
            },
            "oracle": oracle,
            "oracle_success_rate": float(oracle["success_rate"]),
            "oracle_task_success_rate_min": float(min(task_rates)) if task_rates else 0.0,
            "basis_path": str(basis_path),
        }
        metrics_by_k[k] = metrics

        if not args.dry_run:
            np.savez_compressed(
                basis_path,
                mu=mu.astype(np.float32),
                B=B.astype(np.float32),
                k=np.int32(k),
                action_slice_start=np.int32(hand_start),
                action_slice_dim=np.int32(hand_dim),
                explained_ratio=explained_ratio.astype(np.float32),
                cumulative_explained_ratio=cum_ratio.astype(np.float32),
                source_dataset=np.asarray(str(dataset_root), dtype=object),
                tasks=np.asarray(tasks, dtype=object),
                created_at=np.asarray(datetime.now().isoformat(), dtype=object),
            )

        print(
            f"[k={k}] cum_var={metrics['cumulative_explained_variance_ratio_k']:.6f} "
            f"rmse={rmse:.6f} oracle={metrics['oracle_success_rate']:.3f} "
            f"min_task={metrics['oracle_task_success_rate_min']:.3f}"
        )

    best_k = _choose_best_k(metrics_by_k, k_values)
    best_basis_path = basis_path_by_k[best_k]
    if args.save_best_alias and not args.dry_run:
        shutil.copyfile(best_basis_path, out_dir / "mustard_intent_hand_pca_best.npz")

    summary = {
        "created_at": datetime.now().isoformat(),
        "dataset_root": str(dataset_root),
        "tasks": tasks,
        "episodes": int(len(episodes)),
        "transitions": int(X.shape[0]),
        "action_dim": 23,
        "action_slice": {"start": hand_start, "dim": hand_dim},
        "k_values": k_values,
        "best_k": int(best_k),
        "best_basis_path": str(best_basis_path),
        "selection_rule": "max oracle success, then max min-task success, then smaller k",
        "post_settle_steps": int(args.post_settle_steps),
        "metrics_by_k": {str(k): metrics_by_k[k] for k in k_values},
    }
    summary_path = out_dir / "mustard_intent_hand_pca_summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] best_k={best_k} summary={summary_path}")


if __name__ == "__main__":
    main()
