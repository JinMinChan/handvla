#!/usr/bin/env python3
"""Collect corrective near-grasp Franka+Allegro mustard pick-and-lift episodes."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from datetime import datetime
import json
import time

import mujoco
import numpy as np
from mujoco import viewer

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
    _interpolate_hand_pose,
    _make_state_vector,
    _normalize_quat,
    _quat_lerp_normalize,
    _quat_mul,
    _quat_to_rot,
    _resolve_arm_targets,
    _rot_to_quat,
    _sample_spawn_pose,
    _set_mustard_pose,
    _step_arm_ik,
)

ANCHOR_APPROACH = "approach"
ANCHOR_PRESHAPE = "preshape"
ANCHOR_CLOSE = "close"
ANCHOR_PHASES = (ANCHOR_APPROACH, ANCHOR_PRESHAPE, ANCHOR_CLOSE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect corrective pick-and-lift episodes that start from noisy near-grasp "
            "anchor states and save only successful recoveries."
        )
    )
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--target-episodes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset/franka_pickandlift_corrective",
        help="Output root directory. Raw episodes are saved in <out-dir>/raw.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="grasp the mustard bottle",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--capture-hz", type=float, default=5.0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)

    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--approach-steps", type=int, default=700)
    parser.add_argument("--preshape-steps", type=int, default=150)
    parser.add_argument("--close-steps", type=int, default=280)
    parser.add_argument("--lift-steps", type=int, default=260)
    parser.add_argument(
        "--hand-trajectory",
        choices=("keti_human", "power_linear", "thumb_opposition", "thumb_o_wrap"),
        default="thumb_o_wrap",
    )
    parser.add_argument(
        "--preshape-grasp-ratio",
        type=float,
        default=0.45,
        help="Grasp interpolation ratio at end of preshape phase [0,1].",
    )

    parser.add_argument(
        "--anchor-phases",
        type=str,
        default="approach,preshape,close",
        help="Comma-separated subset of: approach,preshape,close.",
    )
    parser.add_argument(
        "--approach-anchor-range",
        type=float,
        nargs=2,
        default=(0.70, 0.98),
        metavar=("LO", "HI"),
        help="Late-approach anchor ratio range.",
    )
    parser.add_argument(
        "--preshape-anchor-range",
        type=float,
        nargs=2,
        default=(0.20, 0.95),
        metavar=("LO", "HI"),
        help="Preshape anchor ratio range.",
    )
    parser.add_argument(
        "--close-anchor-range",
        type=float,
        nargs=2,
        default=(0.05, 0.35),
        metavar=("LO", "HI"),
        help="Early-close anchor ratio range.",
    )

    parser.add_argument(
        "--arm-joint-noise-std",
        type=float,
        default=0.045,
        help="Gaussian std for Franka joint perturbation at anchor (rad).",
    )
    parser.add_argument(
        "--arm-joint-noise-clip",
        type=float,
        default=0.10,
        help="Absolute clip for Franka joint perturbation (rad).",
    )
    parser.add_argument(
        "--hand-joint-noise-std",
        type=float,
        default=0.05,
        help="Gaussian std for Allegro joint perturbation at anchor (rad).",
    )
    parser.add_argument(
        "--hand-joint-noise-clip",
        type=float,
        default=0.12,
        help="Absolute clip for Allegro joint perturbation (rad).",
    )
    parser.add_argument(
        "--object-noise-xy",
        type=float,
        default=0.012,
        help="Uniform object XY perturbation applied at anchor (m).",
    )
    parser.add_argument(
        "--object-noise-z",
        type=float,
        default=0.003,
        help="Uniform object Z perturbation applied at anchor (m).",
    )
    parser.add_argument(
        "--object-yaw-noise-deg",
        type=float,
        default=8.0,
        help="Uniform object yaw perturbation applied at anchor (deg).",
    )

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
        help="Keep mustard fixed during settle/approach/preshape using the current lock pose.",
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
    parser.add_argument("--lift-hold-seconds", type=float, default=1.5)

    parser.add_argument("--min-contacts", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=3)
    parser.add_argument(
        "--require-thumb-contact",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-require-thumb-contact",
        dest="require_thumb_contact",
        action="store_false",
    )
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--max-force", type=float, default=1000.0)
    parser.add_argument("--stable-steps", type=int, default=3)

    parser.add_argument("--viewer", dest="viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.set_defaults(viewer=False)
    parser.add_argument("--viewer-step", type=float, default=60.0)
    parser.add_argument("--viewer-step-delay", type=float, default=None)
    parser.add_argument("--keep-open", action="store_true")
    return parser.parse_args()


def _parse_anchor_phases(text: str) -> tuple[str, ...]:
    phases: list[str] = []
    for token in text.split(","):
        phase = token.strip().lower()
        if not phase:
            continue
        if phase not in ANCHOR_PHASES:
            raise ValueError(f"Unsupported anchor phase: {phase}")
        if phase not in phases:
            phases.append(phase)
    if not phases:
        raise ValueError("At least one anchor phase must be provided.")
    return tuple(phases)


def _clip_range_pair(values: tuple[float, float]) -> tuple[float, float]:
    lo = float(np.clip(values[0], 0.0, 0.999))
    hi = float(np.clip(values[1], 0.0, 0.999))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _sample_anchor_index(
    rng: np.random.Generator,
    total_steps: int,
    ratio_range: tuple[float, float],
) -> int:
    if total_steps <= 1:
        return 0
    lo, hi = _clip_range_pair(ratio_range)
    lo_idx = int(np.floor(lo * total_steps))
    hi_idx = int(np.floor(hi * total_steps))
    lo_idx = int(np.clip(lo_idx, 0, total_steps - 1))
    hi_idx = int(np.clip(hi_idx, lo_idx, total_steps - 1))
    if hi_idx == lo_idx:
        return lo_idx
    return int(rng.integers(lo_idx, hi_idx + 1))


def _phase_name(phase_id: int) -> str:
    return {
        PHASE_SETTLE: "settle",
        PHASE_APPROACH: "approach",
        PHASE_PRESHAPE: "preshape",
        PHASE_CLOSE: "close",
        PHASE_LIFT: "lift",
        PHASE_LIFT_HOLD: "lift_hold",
    }.get(int(phase_id), f"phase_{phase_id}")


def _phase_uses_lock(phase_id: int) -> bool:
    return phase_id in (PHASE_SETTLE, PHASE_APPROACH, PHASE_PRESHAPE)


def _yaw_quat(delta_rad: float) -> np.ndarray:
    return np.asarray(
        [np.cos(0.5 * delta_rad), 0.0, 0.0, np.sin(0.5 * delta_rad)],
        dtype=np.float64,
    )


def _apply_corrective_noise(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_cfg,
    hand_cfg,
    mustard_cfg,
    args: argparse.Namespace,
) -> dict:
    arm_noise = rng.normal(0.0, args.arm_joint_noise_std, size=arm_cfg.qpos_ids.shape[0])
    arm_noise = np.clip(arm_noise, -args.arm_joint_noise_clip, args.arm_joint_noise_clip)
    hand_noise = rng.normal(0.0, args.hand_joint_noise_std, size=hand_cfg.qpos_ids.shape[0])
    hand_noise = np.clip(hand_noise, -args.hand_joint_noise_clip, args.hand_joint_noise_clip)

    data.qpos[arm_cfg.qpos_ids] = np.clip(
        data.qpos[arm_cfg.qpos_ids] + arm_noise,
        arm_cfg.q_min,
        arm_cfg.q_max,
    )
    data.qpos[hand_cfg.qpos_ids] = np.clip(
        data.qpos[hand_cfg.qpos_ids] + hand_noise,
        hand_cfg.q_min,
        hand_cfg.q_max,
    )

    object_pos = data.qpos[mustard_cfg.qpos_adr : mustard_cfg.qpos_adr + 3].copy()
    object_quat = data.qpos[mustard_cfg.qpos_adr + 3 : mustard_cfg.qpos_adr + 7].copy()
    object_delta = np.asarray(
        [
            rng.uniform(-args.object_noise_xy, args.object_noise_xy),
            rng.uniform(-args.object_noise_xy, args.object_noise_xy),
            rng.uniform(-args.object_noise_z, args.object_noise_z),
        ],
        dtype=np.float64,
    )
    yaw_delta = np.deg2rad(float(rng.uniform(-args.object_yaw_noise_deg, args.object_yaw_noise_deg)))
    object_pos = object_pos + object_delta
    object_quat = _normalize_quat(_quat_mul(_yaw_quat(yaw_delta), object_quat))
    _set_mustard_pose(data, mustard_cfg, object_pos, object_quat)

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    return {
        "arm_joint_noise": arm_noise.astype(np.float32),
        "hand_joint_noise": hand_noise.astype(np.float32),
        "object_pos_delta": object_delta.astype(np.float32),
        "object_yaw_delta_deg": float(np.rad2deg(yaw_delta)),
        "record_start_object_qpos": np.concatenate([object_pos, object_quat], axis=0).astype(np.float32),
    }


def _save_corrective_episode_npz(
    episode_path: Path,
    images: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    phases: list[int],
    contacts: list[np.ndarray],
    arm_cmd_pose_wxyz: list[np.ndarray],
    arm_obs_pose_wxyz: list[np.ndarray],
    arm_pose_error: list[np.ndarray],
    success: bool,
    instruction: str,
    side: str,
    control_hz: float,
    capture_hz: float,
    object_qpos: np.ndarray,
    criteria: dict,
    metrics: dict,
    corrective_meta: dict,
) -> None:
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": np.asarray(images, dtype=np.uint8),
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "phase": np.asarray(phases, dtype=np.int32),
        "contact": np.asarray(contacts, dtype=np.float32),
        "arm_cmd_pose_wxyz": np.asarray(arm_cmd_pose_wxyz, dtype=np.float32),
        "arm_obs_pose_wxyz": np.asarray(arm_obs_pose_wxyz, dtype=np.float32),
        "arm_pose_error": np.asarray(arm_pose_error, dtype=np.float32),
        "success": np.asarray(success, dtype=np.bool_),
        "language_instruction": np.asarray(instruction),
        "side": np.asarray(side),
        "control_hz": np.asarray(control_hz, dtype=np.float32),
        "capture_hz": np.asarray(capture_hz, dtype=np.float32),
        "object_qpos": np.asarray(object_qpos, dtype=np.float32),
        "criteria_json": np.asarray(json.dumps(criteria), dtype=object),
        "metrics_json": np.asarray(json.dumps(metrics), dtype=object),
        "action_semantics": np.asarray("joint23_absolute_target_object6d_ik", dtype=object),
        "corrective_anchor_phase": np.asarray(corrective_meta["anchor_phase"]),
        "corrective_anchor_step": np.asarray(corrective_meta["anchor_step"], dtype=np.int32),
        "corrective_anchor_ratio": np.asarray(corrective_meta["anchor_ratio"], dtype=np.float32),
        "corrective_noise_json": np.asarray(json.dumps(corrective_meta["noise"]), dtype=object),
        "saved_at": np.asarray(datetime.now().isoformat()),
    }
    np.savez_compressed(episode_path, **payload)


def run_collection(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    output_root = Path(args.out_dir)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    anchor_phases = _parse_anchor_phases(args.anchor_phases)

    mjcf = franka_allegro_mjcf.load(side=args.side, add_mustard=True)
    model = mjcf.compile()
    data = mujoco.MjData(model)

    control_target_dt = 1.0 / max(float(args.control_hz), 1e-6)
    control_nstep = max(1, int(round(control_target_dt / model.opt.timestep)))
    effective_control_dt = control_nstep * model.opt.timestep
    effective_control_hz = 1.0 / max(effective_control_dt, 1e-9)

    capture_every = _capture_every(effective_control_hz, args.capture_hz)
    effective_capture_hz = effective_control_hz / capture_every

    arm_cfg = _build_arm_config(model, args.side)
    hand_cfg = _build_hand_config(model, args.side)
    mustard_cfg = _build_mustard_config(model)
    contact_cfg = _build_contact_config(model, args.side, mustard_cfg.body_id)

    open_pose = np.clip(np.zeros(16, dtype=np.float32), hand_cfg.q_min, hand_cfg.q_max)
    preshape_ratio = float(np.clip(args.preshape_grasp_ratio, 0.05, 0.95))

    renderer = mujoco.Renderer(model, width=args.image_width, height=args.image_height)
    viewer_ctx = None
    viewer_delay = (
        args.viewer_step_delay
        if args.viewer_step_delay is not None
        else (0.0 if args.viewer_step <= 0 else 1.0 / args.viewer_step)
    )
    capture_cam = None
    if args.viewer:
        viewer_ctx = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True)
        viewer_ctx.__enter__()
        set_default_franka_allegro_camera(viewer_ctx.cam)
    else:
        capture_cam = mujoco.MjvCamera()
        set_default_franka_allegro_camera(capture_cam)

    force_buf = np.zeros(6, dtype=float)
    base_spawn_pos = np.asarray(args.spawn_pos, dtype=np.float32)
    base_spawn_quat = np.asarray(args.spawn_quat, dtype=np.float32)
    approach_offset = np.asarray(args.arm_approach_offset, dtype=np.float32)
    push_offset = np.asarray(args.arm_push_offset, dtype=np.float32)
    approach_rot_quat = _normalize_quat(np.asarray(args.arm_approach_rot_quat, dtype=np.float64))
    push_rot_quat = _normalize_quat(np.asarray(args.arm_push_rot_quat, dtype=np.float64))

    criteria = {
        "arm_reach_threshold": float(args.arm_reach_threshold),
        "ik_rot_gain": float(args.ik_rot_gain),
        "ik_rot_weight": float(args.ik_rot_weight),
        "ik_max_joint_step": float(args.ik_max_joint_step),
        "min_contacts": int(args.min_contacts),
        "min_contact_fingers": int(args.min_contact_fingers),
        "require_thumb_contact": bool(args.require_thumb_contact),
        "min_force": float(args.min_force),
        "max_force": float(args.max_force),
        "stable_steps": int(args.stable_steps),
        "lift_success_delta": float(args.lift_success_delta),
        "lift_hold_seconds": float(args.lift_hold_seconds),
    }

    saved_success = 0
    attempts = 0
    anchor_counts = {phase: 0 for phase in ANCHOR_PHASES}

    try:
        print(
            f"[corrective] target={args.target_episodes} max_attempts={args.max_attempts} "
            f"anchors={','.join(anchor_phases)} control_hz={effective_control_hz:.1f} "
            f"capture_hz={effective_capture_hz:.1f}(every={capture_every}) out_dir={output_root}"
        )

        while saved_success < args.target_episodes and attempts < args.max_attempts:
            attempts += 1
            initial_state = model.key("initial_state").id
            mujoco.mj_resetDataKeyframe(model, data, initial_state)
            mujoco.mj_forward(model, data)

            spawn_pos, spawn_quat = _sample_spawn_pose(base_spawn_pos, base_spawn_quat, rng, args)
            _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
            mujoco.mj_forward(model, data)

            images: list[np.ndarray] = []
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            phases: list[int] = []
            contacts: list[np.ndarray] = []
            arm_cmd_pose_wxyz: list[np.ndarray] = []
            arm_obs_pose_wxyz: list[np.ndarray] = []
            arm_pose_error: list[np.ndarray] = []

            step_counter = 0
            best_contacts = 0
            best_force = 0.0
            best_fingers: set[str] = set()
            arm_min_err = np.inf
            arm_min_rot_err = np.inf
            grasp_stable_hits = 0
            lift_stable_hits = 0
            grasp_acquired = False
            lift_acquired = False
            first_grasp_step = -1
            first_lift_step = -1
            anchor_phase = rng.choice(anchor_phases)
            anchor_counts[anchor_phase] += 1

            locked_object_pos = spawn_pos.astype(np.float64).copy()
            locked_object_quat = spawn_quat.astype(np.float64).copy()

            def _record_and_step(
                hand_cmd: np.ndarray,
                arm_target: ArmTargetPose | None,
                phase_id: int,
                record: bool,
            ) -> dict:
                nonlocal step_counter
                nonlocal best_contacts
                nonlocal best_force
                nonlocal best_fingers
                nonlocal arm_min_err
                nonlocal arm_min_rot_err

                if args.lock_object_until_close and _phase_uses_lock(phase_id):
                    _set_mustard_pose(data, mustard_cfg, locked_object_pos, locked_object_quat)
                    mujoco.mj_forward(model, data)

                n_pre, f_pre, touched_pre = _detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )

                ee_pos_pre = data.xpos[arm_cfg.palm_body_id].copy()
                ee_quat_pre = _rot_to_quat(data.xmat[arm_cfg.palm_body_id].reshape(3, 3))

                if arm_target is None:
                    arm_cmd = data.qpos[arm_cfg.qpos_ids].copy().astype(np.float32)
                    arm_err = 0.0
                    arm_rot_err_deg = 0.0
                    cmd_pose = np.concatenate([ee_pos_pre, ee_quat_pre], axis=0).astype(np.float32)
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
                    cmd_pose = np.concatenate([arm_target.pos, arm_target.quat_wxyz], axis=0).astype(
                        np.float32
                    )

                if record and step_counter % capture_every == 0:
                    thumb_pre = 1.0 if "th" in touched_pre else 0.0
                    contact_stats = np.asarray(
                        [float(n_pre), float(f_pre), float(len(touched_pre)), thumb_pre],
                        dtype=np.float32,
                    )
                    cam = viewer_ctx.cam if viewer_ctx is not None else capture_cam
                    images.append(_capture_frame(renderer, data, cam))
                    states.append(_make_state_vector(data, arm_cfg, hand_cfg, mustard_cfg, contact_stats))
                    actions.append(np.concatenate([arm_cmd, hand_cmd], axis=0).astype(np.float32))
                    phases.append(int(phase_id))
                    contacts.append(contact_stats.copy())
                    arm_cmd_pose_wxyz.append(cmd_pose.copy())
                    arm_obs_pose_wxyz.append(
                        np.concatenate([ee_pos_pre, ee_quat_pre], axis=0).astype(np.float32)
                    )
                    arm_pose_error.append(
                        np.asarray([float(arm_err), float(arm_rot_err_deg)], dtype=np.float32)
                    )

                data.ctrl[arm_cfg.act_ids] = arm_cmd
                data.ctrl[7:23] = hand_cmd
                if args.lock_object_until_close and _phase_uses_lock(phase_id):
                    _set_mustard_pose(data, mustard_cfg, locked_object_pos, locked_object_quat)
                mujoco.mj_step(model, data, nstep=control_nstep)

                n_post, f_post, touched_post = _detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                if record:
                    best_contacts = max(best_contacts, int(n_post))
                    best_force = max(best_force, float(f_post))
                    if len(touched_post) >= len(best_fingers):
                        best_fingers = set(touched_post)

                if arm_target is not None and record:
                    arm_min_err = min(arm_min_err, float(arm_err))
                    arm_min_rot_err = min(arm_min_rot_err, float(arm_rot_err_deg))

                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    if viewer_delay > 0.0:
                        time.sleep(viewer_delay)

                if record:
                    step_counter += 1
                return {
                    "arm_err": float(arm_err),
                    "arm_rot_err_deg": float(arm_rot_err_deg),
                    "n_contacts": int(n_post),
                    "force": float(f_post),
                    "touched": touched_post,
                }

            def _current_targets() -> tuple[ArmTargetPose, ArmTargetPose]:
                return _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=push_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=push_rot_quat,
                    offset_frame=args.arm_offset_frame,
                    rot_frame=args.arm_rot_frame,
                )

            def _run_settle(record: bool) -> None:
                for _ in range(args.settle_steps):
                    _record_and_step(open_pose, None, PHASE_SETTLE, record=record)

            def _run_approach(start_idx: int, record: bool) -> None:
                for _ in range(start_idx, args.approach_steps):
                    approach_target, _ = _current_targets()
                    _record_and_step(open_pose, approach_target, PHASE_APPROACH, record=record)

            def _run_preshape(start_idx: int, record: bool) -> None:
                for i in range(start_idx, args.preshape_steps):
                    alpha = (i + 1) / max(args.preshape_steps, 1)
                    grasp_value = preshape_ratio * alpha
                    hand_cmd = _interpolate_hand_pose(
                        grasp_value=grasp_value,
                        hand_cfg=hand_cfg,
                        hand_trajectory=args.hand_trajectory,
                        preshape_pivot=preshape_ratio,
                    )
                    approach_target, push_target = _current_targets()
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
                    _record_and_step(hand_cmd, arm_target, PHASE_PRESHAPE, record=record)

            def _run_close(start_idx: int, record: bool) -> None:
                nonlocal grasp_stable_hits
                nonlocal grasp_acquired
                nonlocal first_grasp_step
                for i in range(start_idx, args.close_steps):
                    alpha = (i + 1) / max(args.close_steps, 1)
                    grasp_value = preshape_ratio + (1.0 - preshape_ratio) * alpha
                    hand_cmd = _interpolate_hand_pose(
                        grasp_value=grasp_value,
                        hand_cfg=hand_cfg,
                        hand_trajectory=args.hand_trajectory,
                        preshape_pivot=preshape_ratio,
                    )
                    _, push_target = _current_targets()
                    push_target_lifted = ArmTargetPose(
                        pos=(
                            push_target.pos
                            + np.array([0.0, 0.0, args.close_z_clearance], dtype=np.float64)
                        ).astype(np.float64),
                        quat_wxyz=push_target.quat_wxyz.copy(),
                    )
                    info = _record_and_step(hand_cmd, push_target_lifted, PHASE_CLOSE, record=record)
                    if record:
                        meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], args)
                        grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                        if grasp_stable_hits >= args.stable_steps and not grasp_acquired:
                            grasp_acquired = True
                            first_grasp_step = step_counter

            def _run_lift_and_hold(record: bool, object_z_ref: float) -> tuple[float, bool, bool]:
                nonlocal grasp_stable_hits
                nonlocal grasp_acquired
                nonlocal first_grasp_step
                nonlocal lift_stable_hits
                nonlocal lift_acquired
                nonlocal first_lift_step

                object_z_max = object_z_ref
                lift_threshold_z = float(object_z_ref + args.lift_success_delta)
                close_pose = _interpolate_hand_pose(
                    grasp_value=1.0,
                    hand_cfg=hand_cfg,
                    hand_trajectory=args.hand_trajectory,
                    preshape_pivot=preshape_ratio,
                )
                _, push_target = _current_targets()

                for i in range(args.lift_steps):
                    alpha = (i + 1) / max(args.lift_steps, 1)
                    push_pos_lifted = (
                        push_target.pos + np.array([0.0, 0.0, args.close_z_clearance], dtype=np.float64)
                    )
                    lift_target = ArmTargetPose(
                        pos=push_pos_lifted + np.array([0.0, 0.0, args.lift_height * alpha], dtype=float),
                        quat_wxyz=push_target.quat_wxyz.copy(),
                    )
                    info = _record_and_step(close_pose, lift_target, PHASE_LIFT, record=record)
                    object_z = float(data.xpos[mustard_cfg.body_id][2])
                    object_z_max = max(object_z_max, object_z)
                    if record:
                        lifted = (object_z - object_z_ref) >= args.lift_success_delta
                        meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], args)
                        grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                        if grasp_stable_hits >= args.stable_steps and not grasp_acquired:
                            grasp_acquired = True
                            first_grasp_step = step_counter
                        lift_ok = lifted and meets
                        lift_stable_hits = lift_stable_hits + 1 if lift_ok else 0

                hold_target = ArmTargetPose(
                    pos=push_target.pos
                    + np.array([0.0, 0.0, args.close_z_clearance + args.lift_height], dtype=float),
                    quat_wxyz=push_target.quat_wxyz.copy(),
                )
                no_drop_during_hold = True
                hold_contact_ok = True
                lift_hold_steps = max(1, int(np.ceil(args.lift_hold_seconds * effective_control_hz)))
                for _ in range(lift_hold_steps):
                    info = _record_and_step(close_pose, hold_target, PHASE_LIFT_HOLD, record=record)
                    object_z = float(data.xpos[mustard_cfg.body_id][2])
                    object_z_max = max(object_z_max, object_z)
                    if record:
                        lifted = object_z >= lift_threshold_z
                        meets = _contact_meets(info["n_contacts"], info["force"], info["touched"], args)
                        if not lifted:
                            no_drop_during_hold = False
                        if not meets:
                            hold_contact_ok = False

                if record and no_drop_during_hold and not lift_acquired:
                    lift_acquired = True
                    first_lift_step = step_counter
                return object_z_max, no_drop_during_hold, hold_contact_ok

            if anchor_phase == ANCHOR_APPROACH:
                anchor_step = _sample_anchor_index(rng, args.approach_steps, tuple(args.approach_anchor_range))
            elif anchor_phase == ANCHOR_PRESHAPE:
                anchor_step = _sample_anchor_index(rng, args.preshape_steps, tuple(args.preshape_anchor_range))
            else:
                anchor_step = _sample_anchor_index(rng, args.close_steps, tuple(args.close_anchor_range))

            _run_settle(record=False)
            if anchor_phase == ANCHOR_APPROACH:
                for _ in range(anchor_step):
                    approach_target, _ = _current_targets()
                    _record_and_step(open_pose, approach_target, PHASE_APPROACH, record=False)
            else:
                _run_approach(start_idx=0, record=False)
                if anchor_phase == ANCHOR_PRESHAPE:
                    for i in range(anchor_step):
                        alpha = (i + 1) / max(args.preshape_steps, 1)
                        grasp_value = preshape_ratio * alpha
                        hand_cmd = _interpolate_hand_pose(
                            grasp_value=grasp_value,
                            hand_cfg=hand_cfg,
                            hand_trajectory=args.hand_trajectory,
                            preshape_pivot=preshape_ratio,
                        )
                        approach_target, push_target = _current_targets()
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
                        _record_and_step(hand_cmd, arm_target, PHASE_PRESHAPE, record=False)
                elif anchor_phase == ANCHOR_CLOSE:
                    _run_preshape(start_idx=0, record=False)
                    for i in range(anchor_step):
                        alpha = (i + 1) / max(args.close_steps, 1)
                        grasp_value = preshape_ratio + (1.0 - preshape_ratio) * alpha
                        hand_cmd = _interpolate_hand_pose(
                            grasp_value=grasp_value,
                            hand_cfg=hand_cfg,
                            hand_trajectory=args.hand_trajectory,
                            preshape_pivot=preshape_ratio,
                        )
                        _, push_target = _current_targets()
                        push_target_lifted = ArmTargetPose(
                            pos=(
                                push_target.pos
                                + np.array([0.0, 0.0, args.close_z_clearance], dtype=np.float64)
                            ).astype(np.float64),
                            quat_wxyz=push_target.quat_wxyz.copy(),
                        )
                        _record_and_step(hand_cmd, push_target_lifted, PHASE_CLOSE, record=False)

            corrective_noise = _apply_corrective_noise(
                rng,
                model,
                data,
                arm_cfg,
                hand_cfg,
                mustard_cfg,
                args,
            )
            if args.lock_object_until_close and anchor_phase in (ANCHOR_APPROACH, ANCHOR_PRESHAPE):
                locked_object_pos = corrective_noise["record_start_object_qpos"][:3].astype(np.float64).copy()
                locked_object_quat = corrective_noise["record_start_object_qpos"][3:7].astype(np.float64).copy()

            object_qpos = corrective_noise["record_start_object_qpos"].copy()
            anchor_ratio = float(anchor_step / max(
                {
                    ANCHOR_APPROACH: args.approach_steps,
                    ANCHOR_PRESHAPE: args.preshape_steps,
                    ANCHOR_CLOSE: args.close_steps,
                }[anchor_phase],
                1,
            ))

            if anchor_phase == ANCHOR_APPROACH:
                _run_approach(start_idx=anchor_step, record=True)
                _run_preshape(start_idx=0, record=True)
                _run_close(start_idx=0, record=True)
            elif anchor_phase == ANCHOR_PRESHAPE:
                arm_min_err = min(
                    arm_min_err,
                    float(
                        np.linalg.norm(
                            _current_targets()[0].pos - data.xpos[arm_cfg.palm_body_id].copy()
                        )
                    ),
                )
                _run_preshape(start_idx=anchor_step, record=True)
                _run_close(start_idx=0, record=True)
            else:
                _, push_target_now = _current_targets()
                arm_min_err = min(
                    arm_min_err,
                    float(np.linalg.norm(push_target_now.pos - data.xpos[arm_cfg.palm_body_id].copy())),
                )
                _run_close(start_idx=anchor_step, record=True)

            object_z_ref = float(data.xpos[mustard_cfg.body_id][2])
            object_z_max, no_drop_during_hold, hold_contact_ok = _run_lift_and_hold(
                record=True,
                object_z_ref=object_z_ref,
            )

            reached = bool(
                anchor_phase != ANCHOR_APPROACH or arm_min_err <= args.arm_reach_threshold
            )
            success = bool(reached and grasp_acquired and lift_acquired and len(images) > 0)

            metrics = {
                "anchor_phase": anchor_phase,
                "anchor_step": int(anchor_step),
                "anchor_ratio": float(anchor_ratio),
                "arm_min_err": None if not np.isfinite(arm_min_err) else float(arm_min_err),
                "arm_min_rot_err_deg": None
                if not np.isfinite(arm_min_rot_err)
                else float(arm_min_rot_err),
                "reached": reached,
                "grasp_acquired": grasp_acquired,
                "lift_acquired": lift_acquired,
                "first_grasp_step": int(first_grasp_step),
                "first_lift_step": int(first_lift_step),
                "best_contacts": int(best_contacts),
                "best_force": float(best_force),
                "best_fingers": sorted(best_fingers),
                "object_z_ref": float(object_z_ref),
                "object_z_max": float(object_z_max),
                "object_dz_max": float(object_z_max - object_z_ref),
                "no_drop_during_hold": bool(no_drop_during_hold),
                "hold_contact_ok": bool(hold_contact_ok),
                "steps": int(step_counter),
                "frames": int(len(images)),
            }

            corrective_meta = {
                "anchor_phase": anchor_phase,
                "anchor_step": int(anchor_step),
                "anchor_ratio": float(anchor_ratio),
                "noise": {
                    "arm_joint_noise": corrective_noise["arm_joint_noise"].tolist(),
                    "hand_joint_noise": corrective_noise["hand_joint_noise"].tolist(),
                    "object_pos_delta": corrective_noise["object_pos_delta"].tolist(),
                    "object_yaw_delta_deg": float(corrective_noise["object_yaw_delta_deg"]),
                },
            }

            if success:
                ep_name = f"episode_{saved_success:05d}.npz"
                ep_path = raw_dir / ep_name
                _save_corrective_episode_npz(
                    episode_path=ep_path,
                    images=images,
                    states=states,
                    actions=actions,
                    phases=phases,
                    contacts=contacts,
                    arm_cmd_pose_wxyz=arm_cmd_pose_wxyz,
                    arm_obs_pose_wxyz=arm_obs_pose_wxyz,
                    arm_pose_error=arm_pose_error,
                    success=success,
                    instruction=args.instruction,
                    side=args.side,
                    control_hz=float(effective_control_hz),
                    capture_hz=float(effective_capture_hz),
                    object_qpos=object_qpos,
                    criteria=criteria,
                    metrics=metrics,
                    corrective_meta=corrective_meta,
                )
                saved_success += 1
                status = "SAVE"
            else:
                status = "SKIP"

            print(
                f"[attempt {attempts:04d}] {status} anchor={anchor_phase}:{anchor_step} "
                f"saved={saved_success}/{args.target_episodes} "
                f"grasp={int(grasp_acquired)} lift={int(lift_acquired)} "
                f"contacts={best_contacts} fingers={','.join(sorted(best_fingers)) or '-'} "
                f"dz_max={metrics['object_dz_max']:.4f} frames={len(images)}"
            )

        if viewer_ctx is not None and args.keep_open:
            print("Viewer is kept open. Close the MuJoCo window to finish.")
            while viewer_ctx.is_running():
                viewer_ctx.sync()
                if viewer_delay > 0.0:
                    time.sleep(viewer_delay)

    finally:
        renderer.close()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] viewer cleanup error: {exc}")

    finished = saved_success >= args.target_episodes
    summary = {
        "finished": finished,
        "saved_success_episodes": int(saved_success),
        "target_episodes": int(args.target_episodes),
        "attempts": int(attempts),
        "max_attempts": int(args.max_attempts),
        "out_dir": str(output_root),
        "raw_dir": str(raw_dir),
        "instruction": args.instruction,
        "side": args.side,
        "control_hz": float(effective_control_hz),
        "control_hz_requested": float(args.control_hz),
        "capture_hz": float(effective_capture_hz),
        "capture_every": int(capture_every),
        "action_semantics": "joint23_absolute_target_object6d_ik",
        "image_shape": [args.image_height, args.image_width, 3],
        "state_dim": 7 + 16 + 7 + 16 + 3 + 3 + 4 + 4,
        "action_dim": 23,
        "collection_mode": "corrective_anchor_recovery",
        "anchor_distribution": anchor_counts,
        "anchor_ranges": {
            "approach": [float(x) for x in args.approach_anchor_range],
            "preshape": [float(x) for x in args.preshape_anchor_range],
            "close": [float(x) for x in args.close_anchor_range],
        },
        "noise": {
            "arm_joint_noise_std": float(args.arm_joint_noise_std),
            "arm_joint_noise_clip": float(args.arm_joint_noise_clip),
            "hand_joint_noise_std": float(args.hand_joint_noise_std),
            "hand_joint_noise_clip": float(args.hand_joint_noise_clip),
            "object_noise_xy": float(args.object_noise_xy),
            "object_noise_z": float(args.object_noise_z),
            "object_yaw_noise_deg": float(args.object_yaw_noise_deg),
        },
        "criteria": criteria,
        "policy": {
            "settle_steps": int(args.settle_steps),
            "approach_steps": int(args.approach_steps),
            "preshape_steps": int(args.preshape_steps),
            "close_steps": int(args.close_steps),
            "lift_steps": int(args.lift_steps),
            "lift_hold_seconds": float(args.lift_hold_seconds),
            "spawn_pos": [float(x) for x in args.spawn_pos],
            "spawn_quat_wxyz": [float(x) for x in args.spawn_quat],
            "spawn_jitter_xy": float(args.spawn_jitter_xy),
            "spawn_yaw_jitter_deg": float(args.spawn_yaw_jitter_deg),
            "lock_object_until_close": bool(args.lock_object_until_close),
            "arm_approach_offset": [float(x) for x in args.arm_approach_offset],
            "arm_push_offset": [float(x) for x in args.arm_push_offset],
            "arm_offset_frame": args.arm_offset_frame,
            "arm_rot_frame": args.arm_rot_frame,
            "arm_approach_rot_quat_wxyz": [float(x) for x in args.arm_approach_rot_quat],
            "arm_push_rot_quat_wxyz": [float(x) for x in args.arm_push_rot_quat],
            "hand_trajectory": args.hand_trajectory,
            "preshape_grasp_ratio": float(preshape_ratio),
            "ik_gain": float(args.ik_gain),
            "ik_rot_gain": float(args.ik_rot_gain),
            "ik_damping": float(args.ik_damping),
            "ik_rot_weight": float(args.ik_rot_weight),
            "ik_max_joint_step": float(args.ik_max_joint_step),
            "close_z_clearance": float(args.close_z_clearance),
            "lift_height": float(args.lift_height),
            "control_nstep": int(control_nstep),
            "seed": int(args.seed),
        },
    }

    summary_path = output_root / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_collection(args)
    status = "DONE" if summary["finished"] else "STOPPED"
    print(
        f"\n[{status}] saved={summary['saved_success_episodes']}/{summary['target_episodes']} "
        f"attempts={summary['attempts']}"
    )


if __name__ == "__main__":
    main()
