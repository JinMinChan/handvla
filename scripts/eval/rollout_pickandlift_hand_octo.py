#!/usr/bin/env python3
"""Rollout Octo hand policy (k-D synergy) in Franka IK pick-and-lift scene."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import time

import imageio.v2 as imageio
import jax
import mujoco
from mujoco import viewer
import numpy as np
from PIL import Image
import tensorflow as tf

from env import franka_allegro_mjcf
from env.viewer_utils import set_default_franka_allegro_camera
from scripts.data.collect_pickandlift_rlds import (
    PHASE_APPROACH,
    PHASE_CLOSE,
    PHASE_LIFT,
    PHASE_LIFT_HOLD,
    PHASE_PRESHAPE,
    PHASE_SETTLE,
    ArmTargetPose,
    _build_arm_config,
    _build_contact_config,
    _build_hand_config,
    _build_mustard_config,
    _capture_every,
    _capture_frame,
    _contact_meets,
    _detect_contact_with_target,
    _interpolate_hand_pose,  # kept for open-hand init behavior parity
    _make_state_vector,
    _normalize_quat,
    _quat_lerp_normalize,
    _resolve_arm_targets,
    _sample_spawn_pose,
    _set_mustard_pose,
    _step_arm_ik,
)

_DEFAULT_OCTO_SRC = "/home/minchan/Downloads/dual_arm_VLA/octo"
if importlib.util.find_spec("octo") is None:
    octo_src = Path(
        (Path.cwd() / ".octo_src_path").read_text(encoding="utf-8").strip()
        if (Path.cwd() / ".octo_src_path").exists()
        else _DEFAULT_OCTO_SRC
    )
    if octo_src.exists():
        sys.path.insert(0, str(octo_src))

from octo.model.octo_model import OctoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pick-and-lift with arm IK + hand-only Octo policy "
            "(synergy latent action)."
        )
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--basis-path", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="mustard_grasp_oxe")
    parser.add_argument(
        "--task",
        type=str,
        default="reach, grasp, and lift the mustard bottle",
    )
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help=(
            "Skip the first N episodes while preserving RNG progression. "
            "Useful for recording a later deterministic success."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--policy-image-size", type=int, default=256)
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=1.0,
        help=(
            "EMA factor for hand target updates. Pick-and-lift hand-k4 data stores "
            "absolute commands, so 1.0 matches the demonstration semantics."
        ),
    )
    parser.add_argument(
        "--policy-repeat",
        type=int,
        default=20,
        help=(
            "Reuse one predicted hand action for N control steps. "
            "Default 20 matches the 5Hz capture rate of the current 100Hz pick-and-lift dataset."
        ),
    )

    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--approach-steps", type=int, default=700)
    parser.add_argument("--preshape-steps", type=int, default=150)
    parser.add_argument("--close-steps", type=int, default=280)
    parser.add_argument("--lift-steps", type=int, default=260)
    parser.add_argument("--lift-hold-seconds", type=float, default=1.5)

    parser.add_argument(
        "--spawn-pos",
        type=float,
        nargs=3,
        default=(0.62, 0.06, 0.82),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--spawn-quat",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
    )
    parser.add_argument("--spawn-jitter-xy", type=float, default=0.0)
    parser.add_argument("--spawn-yaw-jitter-deg", type=float, default=0.0)

    parser.add_argument(
        "--arm-approach-offset",
        type=float,
        nargs=3,
        default=(-0.09, -0.015, 0.04),
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument(
        "--arm-push-offset",
        type=float,
        nargs=3,
        default=(-0.078, -0.015, 0.01),
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument("--arm-offset-frame", choices=("object", "world"), default="object")
    parser.add_argument("--arm-rot-frame", choices=("object", "world"), default="object")
    parser.add_argument(
        "--arm-approach-rot-quat",
        type=float,
        nargs=4,
        default=(0.70710678, 0.70710678, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
    )
    parser.add_argument(
        "--arm-push-rot-quat",
        type=float,
        nargs=4,
        default=(0.70710678, 0.70710678, 0.0, 0.0),
        metavar=("W", "X", "Y", "Z"),
    )
    parser.add_argument(
        "--lock-object-until-close",
        action="store_true",
        default=True,
        help="Clamp mustard freejoint pose until close phase starts (default: on).",
    )
    parser.add_argument(
        "--no-lock-object-until-close",
        dest="lock_object_until_close",
        action="store_false",
    )

    parser.add_argument("--ik-gain", type=float, default=0.95)
    parser.add_argument("--ik-rot-gain", type=float, default=0.9)
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument("--ik-rot-weight", type=float, default=0.20)
    parser.add_argument("--ik-max-joint-step", type=float, default=0.08)
    parser.add_argument("--close-z-clearance", type=float, default=0.015)
    parser.add_argument("--arm-reach-threshold", type=float, default=0.05)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--lift-success-delta", type=float, default=0.08)

    parser.add_argument("--min-contacts", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=3)
    parser.add_argument(
        "--require-thumb-contact",
        action="store_true",
        default=True,
        help="Require thumb contact for success (default: on).",
    )
    parser.add_argument(
        "--no-require-thumb-contact",
        dest="require_thumb_contact",
        action="store_false",
    )
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--max-force", type=float, default=1000.0)
    parser.add_argument("--stable-steps", type=int, default=3)

    parser.add_argument(
        "--viewer",
        dest="viewer",
        action="store_true",
        help="Show interactive viewer.",
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        help="Run headless without viewer.",
    )
    parser.set_defaults(viewer=False)
    parser.add_argument("--viewer-step", type=float, default=60.0)
    parser.add_argument("--viewer-step-delay", type=float, default=None)
    parser.add_argument("--keep-open", action="store_true")

    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-path", type=str, default="")
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-fps", type=int, default=60)
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def _default_record_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs") / f"pickandlift_hand_octo_rollout_{ts}.mp4"


def _default_summary_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs") / f"pickandlift_hand_octo_eval_{ts}.json"


def _load_basis(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "mu" not in data or "B" not in data:
        raise KeyError(f"Basis file missing mu/B: {path}")
    mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
    B = np.asarray(data["B"], dtype=np.float32)
    if mu.shape[0] != 16:
        raise ValueError(f"Expected mu length 16, got {mu.shape}")
    if B.ndim != 2 or B.shape[0] != 16:
        raise ValueError(f"Expected B shape (16,k), got {B.shape}")
    return mu, B


def _find_stats(model: OctoModel, dataset_name: str) -> tuple[dict | None, dict | None]:
    stats = model.dataset_statistics
    if not isinstance(stats, dict):
        return None, None
    if "action" in stats:
        return stats.get("action"), stats.get("proprio")
    if dataset_name in stats and isinstance(stats[dataset_name], dict):
        ds = stats[dataset_name]
        return ds.get("action"), ds.get("proprio")
    first_key = next(iter(stats.keys()), None)
    if first_key is None or not isinstance(stats[first_key], dict):
        return None, None
    ds = stats[first_key]
    return ds.get("action"), ds.get("proprio")


@dataclass
class EpisodeStats:
    best_contacts: int = 0
    best_force: float = 0.0
    best_fingers: set[str] | None = None
    approach_min_err: float = np.inf
    approach_min_rot_err: float = np.inf
    grasp_stable_hits: int = 0
    lift_stable_hits: int = 0
    grasp_acquired: bool = False
    lift_acquired: bool = False
    first_grasp_step: int = -1
    first_lift_step: int = -1
    object_z_ref: float = 0.0
    object_z_max: float = -1e9
    lift_threshold_z: float = 0.0
    step_counter: int = 0
    no_drop_during_hold: bool = True
    hold_contact_ok: bool = True
    decision_steps: int = 0
    mean_z_norm_acc: float = 0.0
    mean_hand_delta_acc: float = 0.0
    nan_action: bool = False
    action_dim_error: bool = False


def run_rollout(args: argparse.Namespace) -> dict:
    tf.config.set_visible_devices([], "GPU")
    rng = np.random.default_rng(args.seed)
    jax_key = jax.random.PRNGKey(args.seed)

    basis_path = Path(args.basis_path).expanduser().resolve()
    mu, B = _load_basis(basis_path)
    k = int(B.shape[1])

    mjcf = franka_allegro_mjcf.load(side=args.side, add_mustard=True)
    model = mjcf.compile()
    data = mujoco.MjData(model)

    control_target_dt = 1.0 / max(float(args.control_hz), 1e-6)
    control_nstep = max(1, int(round(control_target_dt / model.opt.timestep)))
    effective_control_dt = control_nstep * model.opt.timestep
    effective_control_hz = 1.0 / max(effective_control_dt, 1e-9)

    arm_cfg = _build_arm_config(model, args.side)
    hand_cfg = _build_hand_config(model, args.side)
    mustard_cfg = _build_mustard_config(model)
    contact_cfg = _build_contact_config(model, args.side, mustard_cfg.body_id)
    force_buf = np.zeros(6, dtype=float)

    policy_model = OctoModel.load_pretrained(args.model_path)
    if policy_model.text_processor is None:
        raise RuntimeError("Loaded model has no text_processor.")
    model_window_size = int(policy_model.example_batch["observation"]["timestep_pad_mask"].shape[1])
    task = {"language_instruction": policy_model.text_processor.encode([args.task])}
    action_stats, proprio_stats = _find_stats(policy_model, args.dataset_name)

    viewer_ctx = None
    viewer_delay = (
        args.viewer_step_delay
        if args.viewer_step_delay is not None
        else (0.0 if args.viewer_step <= 0 else 1.0 / args.viewer_step)
    )
    if args.viewer:
        viewer_ctx = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True)
        viewer_ctx.__enter__()
        set_default_franka_allegro_camera(viewer_ctx.cam)

    capture_cam = None
    if viewer_ctx is None:
        capture_cam = mujoco.MjvCamera()
        set_default_franka_allegro_camera(capture_cam)

    renderer_obs = mujoco.Renderer(model, width=args.image_width, height=args.image_height)
    # Policy observations must use the same fixed camera used during dataset collection.
    policy_cam = mujoco.MjvCamera()
    set_default_franka_allegro_camera(policy_cam)
    record_renderer = None
    writer = None
    record_path = None
    if args.record:
        record_path = Path(args.record_path) if args.record_path else _default_record_path()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_renderer = mujoco.Renderer(model, width=args.record_width, height=args.record_height)
        writer = imageio.get_writer(
            record_path,
            fps=args.record_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        print(
            f"[record] writing {record_path} "
            f"({args.record_width}x{args.record_height}@{args.record_fps}fps)",
            flush=True,
        )

    base_spawn_pos = np.asarray(args.spawn_pos, dtype=np.float32)
    base_spawn_quat = np.asarray(args.spawn_quat, dtype=np.float32)
    approach_offset = np.asarray(args.arm_approach_offset, dtype=np.float32)
    push_offset = np.asarray(args.arm_push_offset, dtype=np.float32)
    approach_rot_quat = _normalize_quat(np.asarray(args.arm_approach_rot_quat, dtype=np.float64))
    push_rot_quat = _normalize_quat(np.asarray(args.arm_push_rot_quat, dtype=np.float64))

    open_pose = _interpolate_hand_pose(0.0, hand_cfg, hand_trajectory="power_linear")
    lift_hold_steps = max(1, int(np.ceil(args.lift_hold_seconds * effective_control_hz)))
    capture_every = _capture_every(effective_control_hz, args.record_fps if args.record else 20.0)
    policy_repeat = max(1, int(args.policy_repeat))

    episodes_data: list[dict] = []
    try:
        start_episode = max(0, int(args.start_episode))
        total_rollouts = start_episode + int(args.episodes)
        for ep in range(total_rollouts):
            keep_episode = ep >= start_episode
            out_ep = ep - start_episode
            initial_state = model.key("initial_state").id
            mujoco.mj_resetDataKeyframe(model, data, initial_state)
            mujoco.mj_forward(model, data)

            spawn_pos, spawn_quat = _sample_spawn_pose(base_spawn_pos, base_spawn_quat, rng, args)
            _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
            mujoco.mj_forward(model, data)

            ep_stats = EpisodeStats(best_fingers=set(), object_z_max=-1e9)
            q_hand_cmd = data.qpos[hand_cfg.qpos_ids].astype(np.float32).copy()
            q_hand_cmd = np.clip(q_hand_cmd, hand_cfg.q_min, hand_cfg.q_max)
            cached_q_target = q_hand_cmd.copy()
            repeat_left = 0
            policy_image_history: list[np.ndarray] = []
            policy_state_history: list[np.ndarray] = []

            def _build_policy_obs(obs_image: np.ndarray, obs_state: np.ndarray) -> dict[str, np.ndarray]:
                policy_image_history.append(obs_image.copy())
                policy_state_history.append(obs_state.copy())
                if len(policy_image_history) > model_window_size:
                    policy_image_history.pop(0)
                    policy_state_history.pop(0)

                valid_len = len(policy_image_history)
                pad_len = max(0, model_window_size - valid_len)
                if pad_len > 0:
                    first_image = policy_image_history[0]
                    first_state = policy_state_history[0]
                    image_stack = [first_image] * pad_len + list(policy_image_history)
                    state_stack = [first_state] * pad_len + list(policy_state_history)
                    timestep_pad_mask = np.array(
                        [[False] * pad_len + [True] * valid_len],
                        dtype=np.bool_,
                    )
                else:
                    image_stack = policy_image_history[-model_window_size:]
                    state_stack = policy_state_history[-model_window_size:]
                    timestep_pad_mask = np.ones((1, model_window_size), dtype=np.bool_)

                return {
                    "image_primary": np.stack(image_stack, axis=0)[None, ...],
                    "proprio": np.stack(state_stack, axis=0)[None, ...],
                    "timestep_pad_mask": timestep_pad_mask,
                }

            def step_with_policy(phase_id: int, arm_target: ArmTargetPose | None) -> dict:
                nonlocal q_hand_cmd, jax_key, cached_q_target, repeat_left
                ep_stats.step_counter += 1

                clamp_pre_grasp = args.lock_object_until_close and phase_id in (
                    PHASE_SETTLE,
                    PHASE_APPROACH,
                    PHASE_PRESHAPE,
                )
                if clamp_pre_grasp:
                    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_forward(model, data)

                if repeat_left <= 0:
                    n_pre, f_pre, touched_pre = _detect_contact_with_target(
                        model, data, contact_cfg, force_buf
                    )
                    thumb_pre = 1.0 if "th" in touched_pre else 0.0
                    contact_stats = np.asarray(
                        [float(n_pre), float(f_pre), float(len(touched_pre)), thumb_pre],
                        dtype=np.float32,
                    )
                    state = _make_state_vector(data, arm_cfg, hand_cfg, mustard_cfg, contact_stats)
                    if proprio_stats is not None:
                        mean = np.asarray(proprio_stats["mean"], dtype=np.float32)
                        std = np.asarray(proprio_stats["std"], dtype=np.float32)
                        state = (state - mean) / np.maximum(std, 1e-6)

                    renderer_obs.update_scene(data, camera=policy_cam)
                    obs_image = renderer_obs.render()
                    obs_image = np.asarray(
                        Image.fromarray(obs_image).resize(
                            (args.policy_image_size, args.policy_image_size), Image.BILINEAR
                        ),
                        dtype=np.uint8,
                    )
                    obs = _build_policy_obs(obs_image, state)

                    jax_key, key = jax.random.split(jax_key)
                    if action_stats is not None:
                        pred = policy_model.sample_actions(
                            obs,
                            task,
                            unnormalization_statistics=action_stats,
                            rng=key,
                        )[0, 0]
                    else:
                        pred = policy_model.sample_actions(obs, task, rng=key)[0, 0]
                    pred = np.asarray(pred, dtype=np.float32)

                    if pred.shape[-1] != k:
                        ep_stats.action_dim_error = True
                        return {}
                    if not np.all(np.isfinite(pred)):
                        ep_stats.nan_action = True
                        return {}

                    cached_q_target = mu + B @ pred
                    cached_q_target = np.clip(cached_q_target, hand_cfg.q_min, hand_cfg.q_max)
                    ep_stats.decision_steps += 1
                    ep_stats.mean_z_norm_acc += float(np.linalg.norm(pred))
                    repeat_left = policy_repeat

                q_hand_target = cached_q_target
                alpha = float(np.clip(args.action_smoothing, 0.0, 1.0))
                prev_hand = q_hand_cmd.copy()
                q_hand_cmd = (1.0 - alpha) * q_hand_cmd + alpha * q_hand_target
                q_hand_cmd = np.clip(q_hand_cmd, hand_cfg.q_min, hand_cfg.q_max)
                ep_stats.mean_hand_delta_acc += float(np.linalg.norm(q_hand_cmd - prev_hand))
                repeat_left -= 1

                if arm_target is None:
                    arm_cmd = data.qpos[arm_cfg.qpos_ids].copy().astype(np.float32)
                    arm_err = 0.0
                    arm_rot_err_deg = 0.0
                else:
                    arm_cmd, arm_err, arm_rot_err_deg = _step_arm_ik(
                        model=model,
                        data=data,
                        arm_cfg=arm_cfg,
                        target=arm_target,
                        gain=args.ik_gain,
                        rot_gain=args.ik_rot_gain,
                        damping=args.ik_damping,
                        rot_weight=args.ik_rot_weight,
                        max_joint_step=args.ik_max_joint_step,
                    )

                data.ctrl[arm_cfg.act_ids] = arm_cmd
                data.ctrl[7:23] = q_hand_cmd
                if clamp_pre_grasp:
                    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_step(model, data, nstep=control_nstep)

                n_post, f_post, touched_post = _detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                ep_stats.best_contacts = max(ep_stats.best_contacts, int(n_post))
                ep_stats.best_force = max(ep_stats.best_force, float(f_post))
                if len(touched_post) >= len(ep_stats.best_fingers or set()):
                    ep_stats.best_fingers = set(touched_post)

                if keep_episode and writer is not None and record_renderer is not None:
                    if (ep_stats.step_counter - 1) % max(1, capture_every) == 0:
                        cam = viewer_ctx.cam if viewer_ctx is not None else capture_cam
                        frame = _capture_frame(record_renderer, data, cam)
                        writer.append_data(frame)

                if keep_episode and viewer_ctx is not None:
                    viewer_ctx.sync()
                if keep_episode and viewer_delay > 0.0:
                    time.sleep(viewer_delay)

                return {
                    "arm_err": float(arm_err),
                    "arm_rot_err_deg": float(arm_rot_err_deg),
                    "n_contacts": int(n_post),
                    "force": float(f_post),
                    "touched": touched_post,
                }

            # Settle phase starts from open-hand initialization, but policy outputs must
            # remain stateful across settle steps to preserve the demonstrated trajectory.
            q_hand_cmd = open_pose.copy()
            for _ in range(args.settle_steps):
                info = step_with_policy(PHASE_SETTLE, None)
                if ep_stats.nan_action or ep_stats.action_dim_error:
                    break
            if ep_stats.nan_action or ep_stats.action_dim_error:
                if keep_episode:
                    episodes_data.append(
                        {
                            "episode": out_ep,
                            "source_episode": ep,
                            "success": False,
                            "nan_action": ep_stats.nan_action,
                            "action_dim_error": ep_stats.action_dim_error,
                        }
                    )
                continue

            approach_target, push_target = _resolve_arm_targets(
                data=data,
                mustard_cfg=mustard_cfg,
                approach_offset=approach_offset,
                push_offset=push_offset,
                approach_rot_quat=approach_rot_quat,
                push_rot_quat=push_rot_quat,
                offset_frame=args.arm_offset_frame,
                rot_frame=args.arm_rot_frame,
            )
            mustard_pos_after_settle = data.xpos[mustard_cfg.body_id].copy()
            ep_stats.object_z_ref = float(mustard_pos_after_settle[2])
            ep_stats.object_z_max = ep_stats.object_z_ref
            ep_stats.lift_threshold_z = ep_stats.object_z_ref + float(args.lift_success_delta)

            for _ in range(args.approach_steps):
                approach_target, _ = _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=push_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=push_rot_quat,
                    offset_frame=args.arm_offset_frame,
                    rot_frame=args.arm_rot_frame,
                )
                info = step_with_policy(PHASE_APPROACH, approach_target)
                if not info:
                    break
                ep_stats.approach_min_err = min(ep_stats.approach_min_err, info["arm_err"])
                ep_stats.approach_min_rot_err = min(
                    ep_stats.approach_min_rot_err, info["arm_rot_err_deg"]
                )
            if ep_stats.nan_action or ep_stats.action_dim_error:
                if keep_episode:
                    episodes_data.append(
                        {
                            "episode": out_ep,
                            "source_episode": ep,
                            "success": False,
                            "nan_action": ep_stats.nan_action,
                            "action_dim_error": ep_stats.action_dim_error,
                        }
                    )
                continue

            for i in range(args.preshape_steps):
                alpha = (i + 1) / max(args.preshape_steps, 1)
                approach_target, push_target = _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=push_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=push_rot_quat,
                    offset_frame=args.arm_offset_frame,
                    rot_frame=args.arm_rot_frame,
                )
                push_target_lifted = ArmTargetPose(
                    pos=(
                        push_target.pos
                        + np.array([0.0, 0.0, args.close_z_clearance], dtype=np.float64)
                    ).astype(np.float64),
                    quat_wxyz=push_target.quat_wxyz.copy(),
                )
                arm_target = ArmTargetPose(
                    pos=((1.0 - alpha) * approach_target.pos + alpha * push_target_lifted.pos).astype(
                        np.float64
                    ),
                    quat_wxyz=_quat_lerp_normalize(
                        approach_target.quat_wxyz, push_target_lifted.quat_wxyz, alpha
                    ),
                )
                _ = step_with_policy(PHASE_PRESHAPE, arm_target)
                if ep_stats.nan_action or ep_stats.action_dim_error:
                    break
            if ep_stats.nan_action or ep_stats.action_dim_error:
                if keep_episode:
                    episodes_data.append(
                        {
                            "episode": out_ep,
                            "source_episode": ep,
                            "success": False,
                            "nan_action": ep_stats.nan_action,
                            "action_dim_error": ep_stats.action_dim_error,
                        }
                    )
                continue

            for _ in range(args.close_steps):
                _, push_target = _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=push_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=push_rot_quat,
                    offset_frame=args.arm_offset_frame,
                    rot_frame=args.arm_rot_frame,
                )
                push_target_lifted = ArmTargetPose(
                    pos=(
                        push_target.pos
                        + np.array([0.0, 0.0, args.close_z_clearance], dtype=np.float64)
                    ).astype(np.float64),
                    quat_wxyz=push_target.quat_wxyz.copy(),
                )
                info = step_with_policy(PHASE_CLOSE, push_target_lifted)
                if not info:
                    break
                meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], args)
                ep_stats.grasp_stable_hits = ep_stats.grasp_stable_hits + 1 if meets else 0
                if ep_stats.grasp_stable_hits >= args.stable_steps and not ep_stats.grasp_acquired:
                    ep_stats.grasp_acquired = True
                    ep_stats.first_grasp_step = int(ep_stats.step_counter)
            if ep_stats.nan_action or ep_stats.action_dim_error:
                if keep_episode:
                    episodes_data.append(
                        {
                            "episode": out_ep,
                            "source_episode": ep,
                            "success": False,
                            "nan_action": ep_stats.nan_action,
                            "action_dim_error": ep_stats.action_dim_error,
                        }
                    )
                continue

            _, push_target = _resolve_arm_targets(
                data=data,
                mustard_cfg=mustard_cfg,
                approach_offset=approach_offset,
                push_offset=push_offset,
                approach_rot_quat=approach_rot_quat,
                push_rot_quat=push_rot_quat,
                offset_frame=args.arm_offset_frame,
                rot_frame=args.arm_rot_frame,
            )
            for i in range(args.lift_steps):
                alpha = (i + 1) / max(args.lift_steps, 1)
                push_pos_lifted = (
                    push_target.pos + np.array([0.0, 0.0, args.close_z_clearance], dtype=np.float64)
                )
                lift_target = ArmTargetPose(
                    pos=push_pos_lifted + np.array([0.0, 0.0, args.lift_height * alpha], dtype=float),
                    quat_wxyz=push_target.quat_wxyz.copy(),
                )
                info = step_with_policy(PHASE_LIFT, lift_target)
                if not info:
                    break

                object_z = float(data.xpos[mustard_cfg.body_id][2])
                ep_stats.object_z_max = max(ep_stats.object_z_max, object_z)
                lifted = (object_z - ep_stats.object_z_ref) >= args.lift_success_delta
                meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], args)

                ep_stats.grasp_stable_hits = ep_stats.grasp_stable_hits + 1 if meets else 0
                if ep_stats.grasp_stable_hits >= args.stable_steps and not ep_stats.grasp_acquired:
                    ep_stats.grasp_acquired = True
                    ep_stats.first_grasp_step = int(ep_stats.step_counter)

                lift_ok = lifted and meets
                ep_stats.lift_stable_hits = ep_stats.lift_stable_hits + 1 if lift_ok else 0
            if ep_stats.nan_action or ep_stats.action_dim_error:
                if keep_episode:
                    episodes_data.append(
                        {
                            "episode": out_ep,
                            "source_episode": ep,
                            "success": False,
                            "nan_action": ep_stats.nan_action,
                            "action_dim_error": ep_stats.action_dim_error,
                        }
                    )
                continue

            hold_target = ArmTargetPose(
                pos=push_target.pos
                + np.array([0.0, 0.0, args.close_z_clearance + args.lift_height], dtype=float),
                quat_wxyz=push_target.quat_wxyz.copy(),
            )
            for _ in range(lift_hold_steps):
                info = step_with_policy(PHASE_LIFT_HOLD, hold_target)
                if not info:
                    break
                object_z = float(data.xpos[mustard_cfg.body_id][2])
                ep_stats.object_z_max = max(ep_stats.object_z_max, object_z)
                lifted = object_z >= ep_stats.lift_threshold_z
                meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], args)
                if not lifted:
                    ep_stats.no_drop_during_hold = False
                if not meets:
                    ep_stats.hold_contact_ok = False

            if ep_stats.no_drop_during_hold and not ep_stats.lift_acquired:
                ep_stats.lift_acquired = True
                ep_stats.first_lift_step = int(ep_stats.step_counter)

            reached = bool(ep_stats.approach_min_err <= args.arm_reach_threshold)
            success = bool(reached and ep_stats.grasp_acquired and ep_stats.lift_acquired)
            mean_z_norm = ep_stats.mean_z_norm_acc / max(ep_stats.decision_steps, 1)
            mean_hand_delta = ep_stats.mean_hand_delta_acc / max(ep_stats.decision_steps, 1)

            if keep_episode:
                ep_result = {
                    "episode": out_ep,
                    "source_episode": ep,
                    "success": success,
                    "nan_action": ep_stats.nan_action,
                    "action_dim_error": ep_stats.action_dim_error,
                    "reached": reached,
                    "grasp_acquired": ep_stats.grasp_acquired,
                    "lift_acquired": ep_stats.lift_acquired,
                    "first_grasp_step": int(ep_stats.first_grasp_step),
                    "first_lift_step": int(ep_stats.first_lift_step),
                    "approach_min_err": float(ep_stats.approach_min_err),
                    "approach_min_rot_err_deg": float(ep_stats.approach_min_rot_err),
                    "best_contacts": int(ep_stats.best_contacts),
                    "best_force": float(ep_stats.best_force),
                    "best_fingers": sorted(ep_stats.best_fingers or set()),
                    "object_z_ref": float(ep_stats.object_z_ref),
                    "object_z_max": float(ep_stats.object_z_max),
                    "object_dz_max": float(ep_stats.object_z_max - ep_stats.object_z_ref),
                    "lift_threshold_z": float(ep_stats.lift_threshold_z),
                    "no_drop_during_hold": bool(ep_stats.no_drop_during_hold),
                    "hold_contact_ok": bool(ep_stats.hold_contact_ok),
                    "decision_steps": int(ep_stats.decision_steps),
                    "sim_steps": int(ep_stats.step_counter),
                    "mean_z_norm": float(mean_z_norm),
                    "mean_hand_delta": float(mean_hand_delta),
                }
                episodes_data.append(ep_result)
                print(
                    f"[episode {ep:03d}] success={int(success)} "
                    f"reach={int(reached)} grasp={int(ep_stats.grasp_acquired)} "
                    f"lift={int(ep_stats.lift_acquired)} contacts={ep_stats.best_contacts} "
                    f"fingers={','.join(sorted(ep_stats.best_fingers or set())) or '-'} "
                    f"dz_max={ep_result['object_dz_max']:.4f} "
                    f"mean_z_norm={mean_z_norm:.5f}",
                    flush=True,
                )

        if viewer_ctx is not None and args.keep_open:
            print("Viewer kept open. Close MuJoCo window to finish.", flush=True)
            while viewer_ctx.is_running():
                viewer_ctx.sync()
                if viewer_delay > 0.0:
                    time.sleep(viewer_delay)

    finally:
        if writer is not None:
            writer.close()
            print("[record] finished", flush=True)
        if record_renderer is not None:
            record_renderer.close()
        renderer_obs.close()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] viewer cleanup error: {exc}", flush=True)

    success_rate = (
        float(np.mean([float(ep.get("success", 0.0)) for ep in episodes_data]))
        if episodes_data
        else 0.0
    )
    summary = {
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "basis_path": str(basis_path),
        "dataset_name": args.dataset_name,
        "task": args.task,
        "episodes": int(args.episodes),
        "start_episode": int(start_episode),
        "success_rate": float(success_rate),
        "synergy_k": int(k),
        "window_size": int(model_window_size),
        "control_hz": float(effective_control_hz),
        "control_hz_requested": float(args.control_hz),
        "action_interface": "hand_synergy_kd_arm_ik",
        "record_path": str(record_path) if record_path is not None else "",
        "episodes_data": episodes_data,
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = run_rollout(args)
    print(
        "\n=== PickAndLift Hand Octo Rollout Summary ===\n"
        f"episodes={summary['episodes']} success_rate={100.0 * summary['success_rate']:.1f}%\n"
        f"model={summary['model_path']}\n"
        f"basis={summary['basis_path']}\n"
        f"synergy_k={summary['synergy_k']}"
    )
    out_path = (
        Path(args.save_json).expanduser().resolve() if args.save_json else _default_summary_path()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved summary: {out_path}")


if __name__ == "__main__":
    main()
