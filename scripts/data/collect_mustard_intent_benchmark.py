#!/usr/bin/env python3
"""Collect scripted mustard interaction benchmark episodes for hand-intent tasks."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import time

import imageio.v2 as imageio
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
    _quat_mul,
    _quat_lerp_normalize,
    _quat_to_rot,
    _resolve_arm_targets,
    _rot_to_quat,
    _sample_spawn_pose,
    _set_mustard_pose,
    _step_arm_ik,
)


TASK_WRAP_AND_LIFT = "wrap_and_lift"
TASK_PINCH_AND_LIFT = "pinch_and_lift"
TASK_HOOK_AND_PULL = "hook_and_pull"
TASK_PUSH_OVER = "push_over"
TASK_ROTATE_IN_PLACE = "rotate_in_place"
TASK_NAMES = (
    TASK_WRAP_AND_LIFT,
    TASK_PINCH_AND_LIFT,
    TASK_HOOK_AND_PULL,
    TASK_PUSH_OVER,
    TASK_ROTATE_IN_PLACE,
)

PHASE_PUSH = 6
PHASE_ROTATE = 7
PHASE_PULL = 8


@dataclass(frozen=True)
class IntentLabel:
    interaction_mode: str
    aperture: str
    closure: str
    opposition: str
    tangential: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    instruction: str
    hand_trajectory: str
    preshape_ratio: float
    spawn_offset_local: tuple[float, float, float]
    approach_offset: tuple[float, float, float]
    interact_offset: tuple[float, float, float]
    approach_rot_quat: tuple[float, float, float, float]
    interact_rot_quat: tuple[float, float, float, float]
    settle_steps: int
    approach_steps: int
    preshape_steps: int
    interaction_steps: int
    close_hold_steps: int
    final_steps: int
    lock_object_until_interaction: bool
    intent: IntentLabel


def _task_spec(task_name: str) -> TaskSpec:
    if task_name == TASK_WRAP_AND_LIFT:
        return TaskSpec(
            name=task_name,
            instruction="wrap and lift the bottle",
            hand_trajectory="thumb_o_wrap",
            preshape_ratio=0.45,
            spawn_offset_local=(0.0, 0.0, 0.0),
            approach_offset=(-0.090, -0.015, 0.040),
            interact_offset=(-0.078, -0.015, 0.010),
            approach_rot_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            interact_rot_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            settle_steps=120,
            approach_steps=700,
            preshape_steps=150,
            interaction_steps=280,
            close_hold_steps=40,
            final_steps=260,
            lock_object_until_interaction=True,
            intent=IntentLabel("wrap", "medium", "high", "high", "low"),
        )
    if task_name == TASK_PINCH_AND_LIFT:
        return TaskSpec(
            name=task_name,
            instruction="pinch and lift the bottle",
            hand_trajectory="pinch_precision",
            preshape_ratio=0.55,
            spawn_offset_local=(0.0, 0.0, 0.0),
            approach_offset=(-0.105, -0.015, 0.025),
            interact_offset=(-0.092, -0.015, -0.015),
            approach_rot_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            interact_rot_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            settle_steps=120,
            approach_steps=700,
            preshape_steps=180,
            interaction_steps=260,
            close_hold_steps=140,
            final_steps=220,
            lock_object_until_interaction=True,
            intent=IntentLabel("pinch", "small", "medium_high", "high", "low"),
        )
    if task_name == TASK_HOOK_AND_PULL:
        return TaskSpec(
            name=task_name,
            instruction="hook and pull the bottle",
            hand_trajectory="hook_pull",
            preshape_ratio=1.0,
            spawn_offset_local=(0.0, 0.0, 0.0),
            approach_offset=(-0.185, -0.056, 0.052),
            interact_offset=(-0.122, -0.056, 0.072),
            approach_rot_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            interact_rot_quat=(0.5, 0.5, 0.5, 0.5),
            settle_steps=120,
            approach_steps=560,
            preshape_steps=90,
            interaction_steps=170,
            close_hold_steps=0,
            final_steps=220,
            lock_object_until_interaction=True,
            intent=IntentLabel("hook_pull", "open", "low_medium", "low", "high"),
        )
    if task_name == TASK_PUSH_OVER:
        return TaskSpec(
            name=task_name,
            instruction="push the bottle over",
            hand_trajectory="index_point_push",
            preshape_ratio=1.0,
            spawn_offset_local=(0.0, 0.0, 0.0),
            approach_offset=(-0.185, -0.008, 0.035),
            interact_offset=(-0.150, -0.008, 0.035),
            approach_rot_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            interact_rot_quat=(0.5, 0.5, 0.5, 0.5),
            settle_steps=120,
            approach_steps=560,
            preshape_steps=80,
            interaction_steps=120,
            close_hold_steps=0,
            final_steps=90,
            lock_object_until_interaction=True,
            intent=IntentLabel("push", "open", "low", "low", "high"),
        )
    if task_name == TASK_ROTATE_IN_PLACE:
        return TaskSpec(
            name=task_name,
            instruction="rotate the bottle",
            hand_trajectory="rotate_wrap",
            preshape_ratio=0.50,
            spawn_offset_local=(0.0, 0.0, 0.0),
            approach_offset=(-0.095, -0.010, 0.045),
            interact_offset=(-0.078, -0.010, 0.020),
            approach_rot_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            interact_rot_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            settle_steps=120,
            approach_steps=620,
            preshape_steps=120,
            interaction_steps=260,
            close_hold_steps=0,
            final_steps=80,
            lock_object_until_interaction=True,
            intent=IntentLabel("rotate", "medium", "low_medium", "medium", "high"),
        )
    raise ValueError(f"Unsupported task: {task_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect scripted mustard interaction episodes for the multi-task "
            "hand-intent benchmark."
        )
    )
    parser.add_argument("--task", choices=TASK_NAMES, default=TASK_WRAP_AND_LIFT)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--target-episodes", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Default: dataset/mustard_intent/<task>",
    )
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--capture-hz", type=float, default=10.0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)

    parser.add_argument("--spawn-pos", type=float, nargs=3, default=(0.78, 0.06, 0.82))
    parser.add_argument("--spawn-quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--spawn-jitter-xy", type=float, default=0.0)
    parser.add_argument("--spawn-yaw-jitter-deg", type=float, default=0.0)
    parser.add_argument(
        "--settle-steps-override",
        type=int,
        default=-1,
        help="If >= 0, override the task's default scripted settle length.",
    )
    parser.add_argument(
        "--arm-disturb-xyz",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        help=(
            "Low-frequency object-frame arm target disturbance amplitude in meters. "
            "Applied only in phases listed by --arm-disturb-phases."
        ),
    )
    parser.add_argument(
        "--arm-disturb-resample-steps",
        type=int,
        default=35,
        help="Control steps between piecewise-constant disturbance resamples.",
    )
    parser.add_argument(
        "--arm-disturb-phases",
        type=str,
        default="approach,preshape",
        help="Comma-separated scripted phases to disturb: settle,approach,preshape,close,lift,push,pull,rotate",
    )

    parser.add_argument("--ik-gain", type=float, default=0.95)
    parser.add_argument("--ik-rot-gain", type=float, default=0.9)
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument("--ik-rot-weight", type=float, default=0.20)
    parser.add_argument("--ik-max-joint-step", type=float, default=0.08)
    parser.add_argument("--close-z-clearance", type=float, default=0.015)

    parser.add_argument("--arm-reach-threshold", type=float, default=0.05)
    parser.add_argument(
        "--approach-early-exit-pos-err",
        type=float,
        default=0.01,
        help="Break approach early when arm position error stays below this threshold.",
    )
    parser.add_argument(
        "--approach-early-exit-rot-err-deg",
        type=float,
        default=2.0,
        help="Break approach early when arm rotation error stays below this threshold.",
    )
    parser.add_argument(
        "--approach-early-exit-stable-steps",
        type=int,
        default=3,
        help="Number of consecutive captured approach frames required for early exit.",
    )
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--lift-success-delta", type=float, default=0.08)
    parser.add_argument("--lift-hold-seconds", type=float, default=1.5)
    parser.add_argument(
        "--post-grasp-upright-steps",
        type=int,
        default=40,
        help=(
            "Control steps spent reorienting the grasped bottle toward the upright "
            "lift pose before any upward motion begins."
        ),
    )
    parser.add_argument(
        "--post-grasp-retract-steps",
        type=int,
        default=25,
        help=(
            "Control steps spent retracting straight up while keeping the realized "
            "grasp orientation fixed before any upright reorientation begins."
        ),
    )
    parser.add_argument(
        "--post-grasp-retract-height",
        type=float,
        default=0.03,
        help=(
            "World-z retract distance executed immediately after grasp acquisition "
            "before upright reorientation and the main lift."
        ),
    )
    parser.add_argument(
        "--lift-upright-rot-fraction",
        type=float,
        default=0.25,
        help=(
            "Fraction of lift duration over which the wrist orientation is steered "
            "from the realized grasp pose toward the upright lift pose. Smaller "
            "values make the wrist become upright earlier in the lift."
        ),
    )
    parser.add_argument(
        "--lift-upright-rot-power",
        type=float,
        default=0.5,
        help=(
            "Exponent applied to the normalized orientation interpolation progress. "
            "Values below 1.0 make the wrist stand up earlier in the lift."
        ),
    )

    parser.add_argument("--min-contacts", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=3)
    parser.add_argument("--require-thumb-contact", action="store_true", default=True)
    parser.add_argument("--no-require-thumb-contact", dest="require_thumb_contact", action="store_false")
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--max-force", type=float, default=1000.0)
    parser.add_argument("--stable-steps", type=int, default=3)

    parser.add_argument("--push-success-tilt-deg", type=float, default=55.0)
    parser.add_argument("--push-max-lift-dz", type=float, default=0.04)
    parser.add_argument("--push-retract-steps", type=int, default=80)
    parser.add_argument(
        "--push-release-steps",
        type=int,
        default=3,
        help=(
            "Consecutive capture frames after toppling with no fingertip-object contact "
            "required to count push_over as released rather than finger-trapped."
        ),
    )
    parser.add_argument("--pull-success-dx", type=float, default=0.08)
    parser.add_argument("--pull-max-lift-dz", type=float, default=0.05)

    parser.add_argument("--rotate-success-yaw-deg", type=float, default=35.0)
    parser.add_argument("--rotate-max-tilt-deg", type=float, default=25.0)
    parser.add_argument("--rotate-max-xy-drift", type=float, default=0.06)
    parser.add_argument("--rotate-arc-deg", type=float, default=70.0)

    parser.add_argument("--viewer", dest="viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.set_defaults(viewer=False)
    parser.add_argument("--show-axes", action="store_true", default=False)
    parser.add_argument("--viewer-step", type=float, default=60.0)
    parser.add_argument("--viewer-step-delay", type=float, default=None)
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-path", type=str, default="")
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-fps", type=int, default=60)
    return parser.parse_args()


def _default_out_dir(task_name: str) -> Path:
    return Path("dataset/mustard_intent") / task_name


def _default_record_path(task_name: str) -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs/videos") / f"mustard_intent_{task_name}_{ts}.mp4"


def _tilt_deg(rot: np.ndarray) -> float:
    z_axis = np.asarray(rot, dtype=np.float64).reshape(3, 3)[:, 2]
    return float(np.rad2deg(np.arccos(np.clip(z_axis[2], -1.0, 1.0))))


def _yaw_rad(rot: np.ndarray) -> float:
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def _angle_diff_deg(a: float, b: float) -> float:
    d = float(np.arctan2(np.sin(a - b), np.cos(a - b)))
    return float(np.rad2deg(abs(d)))


def _pinch_contact_meets(
    n_contacts: int,
    total_force: float,
    touched_fingers: set[str],
    args: argparse.Namespace,
) -> bool:
    if "th" not in touched_fingers or "ff" not in touched_fingers:
        return False
    return (
        n_contacts >= 2
        and total_force >= args.min_force
        and total_force <= args.max_force
    )


def _push_contact_meets(
    n_contacts: int,
    total_force: float,
    touched_fingers: set[str],
    args: argparse.Namespace,
) -> bool:
    return (
        "ff" in touched_fingers
        and len(touched_fingers) <= 2
        and n_contacts >= 1
        and total_force >= args.min_force
        and total_force <= args.max_force
    )


def _pull_contact_meets(
    n_contacts: int,
    total_force: float,
    touched_fingers: set[str],
    args: argparse.Namespace,
) -> bool:
    return (
        "ff" in touched_fingers
        and n_contacts >= 1
        and total_force >= args.min_force
        and total_force <= args.max_force
    )


def _save_episode_npz(
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
    task_name: str,
    intent: IntentLabel,
    side: str,
    control_hz: float,
    capture_hz: float,
    object_qpos: np.ndarray,
    criteria: dict,
    metrics: dict,
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
        "task_name": np.asarray(task_name),
        "intent_json": np.asarray(json.dumps(intent.__dict__), dtype=object),
        "side": np.asarray(side),
        "control_hz": np.asarray(control_hz, dtype=np.float32),
        "capture_hz": np.asarray(capture_hz, dtype=np.float32),
        "object_qpos": np.asarray(object_qpos, dtype=np.float32),
        "criteria_json": np.asarray(json.dumps(criteria), dtype=object),
        "metrics_json": np.asarray(json.dumps(metrics), dtype=object),
        "action_semantics": np.asarray("joint23_absolute_target_object6d_ik", dtype=object),
        "saved_at": np.asarray(datetime.now().isoformat()),
    }
    np.savez_compressed(episode_path, **payload)


def run_collection(args: argparse.Namespace) -> dict:
    spec = _task_spec(args.task)
    instruction = args.instruction or spec.instruction
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(args.task)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    mjcf = franka_allegro_mjcf.load(
        side=args.side,
        add_mustard=True,
        add_frame_axes=args.show_axes,
    )
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

    renderer = mujoco.Renderer(model, width=args.image_width, height=args.image_height)
    record_renderer = None
    writer = None
    record_path = None
    frame_dt = 0.0
    next_frame_time = 0.0
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

    if args.record:
        record_path = Path(args.record_path) if args.record_path else _default_record_path(args.task)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            record_path,
            fps=args.record_fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        record_renderer = mujoco.Renderer(
            model,
            width=args.record_width,
            height=args.record_height,
        )
        frame_dt = 1.0 / max(int(args.record_fps), 1)
        next_frame_time = 0.0
        print(
            f"[record] writing {record_path} "
            f"({args.record_width}x{args.record_height}@{args.record_fps}fps)",
            flush=True,
        )

    force_buf = np.zeros(6, dtype=float)
    base_spawn_pos = np.asarray(args.spawn_pos, dtype=np.float32)
    base_spawn_quat = np.asarray(args.spawn_quat, dtype=np.float32)
    spawn_offset_local = np.asarray(spec.spawn_offset_local, dtype=np.float32)
    approach_offset = np.asarray(spec.approach_offset, dtype=np.float32)
    interact_offset = np.asarray(spec.interact_offset, dtype=np.float32)
    approach_rot_quat = _normalize_quat(np.asarray(spec.approach_rot_quat, dtype=np.float64))
    interact_rot_quat = _normalize_quat(np.asarray(spec.interact_rot_quat, dtype=np.float64))
    preshape_ratio = float(np.clip(spec.preshape_ratio, 0.05, 0.95))
    settle_steps = spec.settle_steps if int(args.settle_steps_override) < 0 else int(args.settle_steps_override)
    arm_disturb_xyz = np.asarray(args.arm_disturb_xyz, dtype=np.float64)
    arm_disturb_resample_steps = max(int(args.arm_disturb_resample_steps), 1)
    disturb_phase_tokens = {
        token.strip().lower()
        for token in str(args.arm_disturb_phases).split(",")
        if token.strip()
    }
    phase_name_map = {
        PHASE_SETTLE: "settle",
        PHASE_APPROACH: "approach",
        PHASE_PRESHAPE: "preshape",
        PHASE_CLOSE: "close",
        PHASE_LIFT: "lift",
        PHASE_LIFT_HOLD: "lift",
        PHASE_PUSH: "push",
        PHASE_PULL: "pull",
        PHASE_ROTATE: "rotate",
    }
    idle_pose = np.clip(np.zeros(16, dtype=np.float32), hand_cfg.q_min, hand_cfg.q_max)

    criteria = {
        "task_name": args.task,
        "arm_reach_threshold": float(args.arm_reach_threshold),
        "min_contacts": int(args.min_contacts),
        "min_contact_fingers": int(args.min_contact_fingers),
        "require_thumb_contact": bool(args.require_thumb_contact),
        "min_force": float(args.min_force),
        "max_force": float(args.max_force),
        "stable_steps": int(args.stable_steps),
        "lift_success_delta": float(args.lift_success_delta),
        "lift_hold_seconds": float(args.lift_hold_seconds),
        "push_success_tilt_deg": float(args.push_success_tilt_deg),
        "push_max_lift_dz": float(args.push_max_lift_dz),
        "pull_success_dx": float(args.pull_success_dx),
        "pull_max_lift_dz": float(args.pull_max_lift_dz),
        "rotate_success_yaw_deg": float(args.rotate_success_yaw_deg),
        "rotate_max_tilt_deg": float(args.rotate_max_tilt_deg),
        "rotate_max_xy_drift": float(args.rotate_max_xy_drift),
    }

    saved_success = 0
    attempts = 0
    last_attempt_metrics: dict | None = None

    try:
        print(
            f"[intent] task={args.task} target={args.target_episodes} max_attempts={args.max_attempts} "
            f"control_hz={effective_control_hz:.1f} capture_hz={effective_capture_hz:.1f} "
            f"out_dir={out_dir}"
        )
        while saved_success < args.target_episodes and attempts < args.max_attempts:
            attempts += 1
            initial_state = model.key("initial_state").id
            mujoco.mj_resetDataKeyframe(model, data, initial_state)
            mujoco.mj_forward(model, data)

            spawn_pos, spawn_quat = _sample_spawn_pose(base_spawn_pos, base_spawn_quat, rng, args)
            if np.any(np.abs(spawn_offset_local) > 0.0):
                spawn_rot = _quat_to_rot(spawn_quat)
                spawn_pos = spawn_pos + (spawn_rot @ spawn_offset_local.astype(np.float64)).astype(
                    np.float32
                )
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
            approach_min_err = np.inf
            approach_min_rot_err = np.inf
            approach_steps_used = 0
            grasp_stable_hits = 0
            task_success_stable_hits = 0
            grasp_acquired = False
            task_success = False
            first_grasp_step = -1
            first_success_step = -1

            object_qpos = np.concatenate([spawn_pos, spawn_quat], axis=0).astype(np.float32)
            lift_hold_steps = max(1, int(np.ceil(args.lift_hold_seconds * effective_control_hz)))
            initial_obj_rot = data.xmat[mustard_cfg.body_id].reshape(3, 3).copy()
            initial_obj_yaw = _yaw_rad(initial_obj_rot)
            object_z_ref = float(data.xpos[mustard_cfg.body_id][2])
            object_z_max = object_z_ref
            object_xy_ref = data.xpos[mustard_cfg.body_id][:2].copy()
            disturb_offset_local = np.zeros(3, dtype=np.float64)
            disturb_countdown = 0

            def _disturb_arm_target(
                arm_target: ArmTargetPose | None,
                phase_id: int,
            ) -> ArmTargetPose | None:
                nonlocal disturb_offset_local, disturb_countdown
                if arm_target is None or not np.any(np.abs(arm_disturb_xyz) > 0.0):
                    return arm_target
                phase_name = phase_name_map.get(int(phase_id), "")
                if phase_name not in disturb_phase_tokens:
                    disturb_offset_local = np.zeros(3, dtype=np.float64)
                    disturb_countdown = 0
                    return arm_target
                if disturb_countdown <= 0:
                    disturb_offset_local = rng.uniform(-arm_disturb_xyz, arm_disturb_xyz).astype(
                        np.float64
                    )
                    disturb_countdown = arm_disturb_resample_steps
                disturb_countdown -= 1
                obj_rot = data.xmat[mustard_cfg.body_id].reshape(3, 3).copy()
                disturb_world = obj_rot @ disturb_offset_local
                return ArmTargetPose(
                    pos=(np.asarray(arm_target.pos, dtype=np.float64) + disturb_world).astype(
                        np.float64
                    ),
                    quat_wxyz=np.asarray(arm_target.quat_wxyz, dtype=np.float64).copy(),
                )

            def _record_and_step(hand_cmd: np.ndarray, arm_target: ArmTargetPose | None, phase_id: int):
                nonlocal step_counter, best_contacts, best_force, best_fingers, next_frame_time

                if spec.lock_object_until_interaction and phase_id in (
                    PHASE_SETTLE,
                    PHASE_APPROACH,
                    PHASE_PRESHAPE,
                ):
                    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_forward(model, data)

                n_pre, f_pre, touched_pre = _detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                ee_pos_pre = data.xpos[arm_cfg.palm_body_id].copy()
                ee_quat_pre = _rot_to_quat(data.xmat[arm_cfg.palm_body_id].reshape(3, 3))

                arm_target = _disturb_arm_target(arm_target, phase_id)

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

                if step_counter % capture_every == 0:
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
                if spec.lock_object_until_interaction and phase_id in (
                    PHASE_SETTLE,
                    PHASE_APPROACH,
                    PHASE_PRESHAPE,
                ):
                    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_step(model, data, nstep=control_nstep)

                n_post, f_post, touched_post = _detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                best_contacts = max(best_contacts, int(n_post))
                best_force = max(best_force, float(f_post))
                if len(touched_post) >= len(best_fingers):
                    best_fingers = set(touched_post)

                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    if viewer_delay > 0.0:
                        time.sleep(viewer_delay)
                if writer is not None and record_renderer is not None:
                    cam = viewer_ctx.cam if viewer_ctx is not None else capture_cam
                    while data.time + 1e-9 >= next_frame_time:
                        record_renderer.update_scene(data, camera=cam)
                        writer.append_data(record_renderer.render())
                        next_frame_time += frame_dt

                step_counter += 1
                return {
                    "arm_err": float(arm_err),
                    "arm_rot_err_deg": float(arm_rot_err_deg),
                    "n_contacts": int(n_post),
                    "force": float(f_post),
                    "touched": touched_post,
                    "object_pos": data.xpos[mustard_cfg.body_id].copy(),
                    "object_rot": data.xmat[mustard_cfg.body_id].reshape(3, 3).copy(),
                }

            for _ in range(settle_steps):
                _record_and_step(idle_pose, None, PHASE_SETTLE)

            approach_target, interact_target = _resolve_arm_targets(
                data=data,
                mustard_cfg=mustard_cfg,
                approach_offset=approach_offset,
                push_offset=interact_offset,
                approach_rot_quat=approach_rot_quat,
                push_rot_quat=interact_rot_quat,
                offset_frame="object",
                rot_frame="object",
            )

            approach_exit_stable_hits = 0
            for _ in range(spec.approach_steps):
                if spec.lock_object_until_interaction:
                    _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                    mujoco.mj_forward(model, data)
                approach_target, interact_target = _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=interact_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=interact_rot_quat,
                    offset_frame="object",
                    rot_frame="object",
                )
                info = _record_and_step(idle_pose, approach_target, PHASE_APPROACH)
                approach_steps_used += 1
                approach_min_err = min(approach_min_err, info["arm_err"])
                approach_min_rot_err = min(approach_min_rot_err, info["arm_rot_err_deg"])
                if (
                    info["arm_err"] <= float(args.approach_early_exit_pos_err)
                    and info["arm_rot_err_deg"] <= float(args.approach_early_exit_rot_err_deg)
                ):
                    approach_exit_stable_hits += 1
                else:
                    approach_exit_stable_hits = 0
                if approach_exit_stable_hits >= max(int(args.approach_early_exit_stable_steps), 1):
                    break

            if args.task in (
                TASK_WRAP_AND_LIFT,
                TASK_PINCH_AND_LIFT,
                TASK_HOOK_AND_PULL,
                TASK_ROTATE_IN_PLACE,
                TASK_PUSH_OVER,
            ):
                for i in range(spec.preshape_steps):
                    alpha = (i + 1) / max(spec.preshape_steps, 1)
                    hand_cmd = _interpolate_hand_pose(
                        grasp_value=preshape_ratio * alpha,
                        hand_cfg=hand_cfg,
                        hand_trajectory=spec.hand_trajectory,
                        preshape_pivot=preshape_ratio,
                    )
                    if spec.lock_object_until_interaction:
                        _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                        mujoco.mj_forward(model, data)
                    approach_target, interact_target = _resolve_arm_targets(
                        data=data,
                        mustard_cfg=mustard_cfg,
                        approach_offset=approach_offset,
                        push_offset=interact_offset,
                        approach_rot_quat=approach_rot_quat,
                        push_rot_quat=interact_rot_quat,
                        offset_frame="object",
                        rot_frame="object",
                    )
                    if args.task == TASK_PUSH_OVER:
                        # Keep a safe stand-off while the hand morphs into the pointing pose,
                        # and align the finger push axis before moving forward.
                        arm_target = ArmTargetPose(
                            pos=approach_target.pos.astype(np.float64),
                            quat_wxyz=_quat_lerp_normalize(
                                approach_target.quat_wxyz,
                                interact_target.quat_wxyz,
                                alpha,
                            ),
                        )
                    elif args.task == TASK_HOOK_AND_PULL:
                        arm_target = ArmTargetPose(
                            pos=approach_target.pos.astype(np.float64),
                            quat_wxyz=_quat_lerp_normalize(
                                approach_target.quat_wxyz,
                                interact_target.quat_wxyz,
                                alpha,
                            ),
                        )
                    else:
                        arm_target = ArmTargetPose(
                            pos=((1.0 - alpha) * approach_target.pos + alpha * interact_target.pos).astype(
                                np.float64
                            ),
                            quat_wxyz=_quat_lerp_normalize(
                                approach_target.quat_wxyz, interact_target.quat_wxyz, alpha
                            ),
                        )
                    _record_and_step(hand_cmd, arm_target, PHASE_PRESHAPE)

            if args.task in (TASK_WRAP_AND_LIFT, TASK_PINCH_AND_LIFT):
                for i in range(spec.interaction_steps):
                    alpha = (i + 1) / max(spec.interaction_steps, 1)
                    grasp_value = preshape_ratio + (1.0 - preshape_ratio) * alpha
                    hand_cmd = _interpolate_hand_pose(
                        grasp_value=grasp_value,
                        hand_cfg=hand_cfg,
                        hand_trajectory=spec.hand_trajectory,
                        preshape_pivot=preshape_ratio,
                    )
                    approach_target, interact_target = _resolve_arm_targets(
                        data=data,
                        mustard_cfg=mustard_cfg,
                        approach_offset=approach_offset,
                        push_offset=interact_offset,
                        approach_rot_quat=approach_rot_quat,
                        push_rot_quat=interact_rot_quat,
                        offset_frame="object",
                        rot_frame="object",
                    )
                    close_target = ArmTargetPose(
                        pos=interact_target.pos.astype(np.float64),
                        quat_wxyz=interact_target.quat_wxyz.copy(),
                    )
                    info = _record_and_step(hand_cmd, close_target, PHASE_CLOSE)
                    if args.task == TASK_PINCH_AND_LIFT:
                        meets = _pinch_contact_meets(
                            info["n_contacts"], info["force"], info["touched"], args
                        )
                    else:
                        meets = _contact_meets(
                            info["n_contacts"], info["force"], info["touched"], args
                        )
                    grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                    if grasp_stable_hits >= args.stable_steps and not grasp_acquired:
                        grasp_acquired = True
                        first_grasp_step = step_counter
                        break

                close_pose = _interpolate_hand_pose(
                    grasp_value=1.0,
                    hand_cfg=hand_cfg,
                    hand_trajectory=spec.hand_trajectory,
                    preshape_pivot=preshape_ratio,
                )
                _, interact_target = _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=interact_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=interact_rot_quat,
                    offset_frame="object",
                    rot_frame="object",
                )
                close_target = ArmTargetPose(
                    pos=interact_target.pos.astype(np.float64),
                    quat_wxyz=interact_target.quat_wxyz.copy(),
                )
                if not grasp_acquired:
                    for _ in range(spec.close_hold_steps):
                        info = _record_and_step(close_pose, close_target, PHASE_CLOSE)
                        if args.task == TASK_PINCH_AND_LIFT:
                            meets = _pinch_contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        else:
                            meets = _contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                        if grasp_stable_hits >= args.stable_steps and not grasp_acquired:
                            grasp_acquired = True
                            first_grasp_step = step_counter
                            break
                no_drop_during_hold = False
                hold_contact_ok = False
                if grasp_acquired:
                    # Lift from the actually achieved grasp pose. This keeps
                    # collection closer to a true world-z raise instead of
                    # re-pulling the arm toward the nominal interact target
                    # before lifting.
                    lift_base_pos = data.xpos[arm_cfg.palm_body_id].copy().astype(np.float64)
                    lift_base_quat = _rot_to_quat(
                        data.xmat[arm_cfg.palm_body_id].reshape(3, 3)
                    ).astype(np.float64)
                    # Keep the bottle's original upright world frame as the
                    # reference for the lift orientation, then smoothly steer
                    # the wrist from the realized grasp pose toward that
                    # upright lift pose during the raise.
                    lift_upright_quat = _normalize_quat(
                        _quat_mul(spawn_quat.astype(np.float64), interact_rot_quat)
                    ).astype(np.float64)
                    retract_steps = max(int(args.post_grasp_retract_steps), 0)
                    retract_height = max(float(args.post_grasp_retract_height), 0.0)
                    retract_pos = lift_base_pos.copy()
                    for i in range(retract_steps):
                        retract_alpha = (i + 1) / max(retract_steps, 1)
                        retract_target = ArmTargetPose(
                            pos=lift_base_pos
                            + np.array(
                                [0.0, 0.0, retract_height * retract_alpha],
                                dtype=np.float64,
                            ),
                            quat_wxyz=lift_base_quat.copy(),
                        )
                        info = _record_and_step(close_pose, retract_target, PHASE_LIFT)
                        retract_pos = retract_target.pos.copy()
                        object_z = float(data.xpos[mustard_cfg.body_id][2])
                        object_z_max = max(object_z_max, object_z)
                        if args.task == TASK_PINCH_AND_LIFT:
                            meets = _pinch_contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        else:
                            meets = _contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                    for i in range(max(int(args.post_grasp_upright_steps), 0)):
                        upright_alpha = (i + 1) / max(int(args.post_grasp_upright_steps), 1)
                        upright_target = ArmTargetPose(
                            pos=retract_pos.copy(),
                            quat_wxyz=_quat_lerp_normalize(
                                lift_base_quat,
                                lift_upright_quat,
                                float(upright_alpha),
                            ),
                        )
                        info = _record_and_step(close_pose, upright_target, PHASE_LIFT)
                        object_z = float(data.xpos[mustard_cfg.body_id][2])
                        object_z_max = max(object_z_max, object_z)
                        if args.task == TASK_PINCH_AND_LIFT:
                            meets = _pinch_contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        else:
                            meets = _contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                    rot_fraction = max(float(args.lift_upright_rot_fraction), 1e-6)
                    rot_power = max(float(args.lift_upright_rot_power), 1e-6)
                    remaining_lift = max(0.0, float(args.lift_height) - retract_height)
                    for i in range(spec.final_steps):
                        alpha = (i + 1) / max(spec.final_steps, 1)
                        rot_alpha = min(1.0, alpha / rot_fraction)
                        rot_alpha = float(rot_alpha**rot_power)
                        lift_target = ArmTargetPose(
                            pos=retract_pos
                            + np.array(
                                [0.0, 0.0, remaining_lift * alpha],
                                dtype=np.float64,
                            ),
                            quat_wxyz=_quat_lerp_normalize(
                                lift_upright_quat,
                                lift_upright_quat,
                                rot_alpha,
                            ),
                        )
                        info = _record_and_step(close_pose, lift_target, PHASE_LIFT)
                        object_z = float(data.xpos[mustard_cfg.body_id][2])
                        object_z_max = max(object_z_max, object_z)
                        lifted = (object_z - object_z_ref) >= args.lift_success_delta
                        if args.task == TASK_PINCH_AND_LIFT:
                            meets = _pinch_contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        else:
                            meets = _contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                        task_success_stable_hits = task_success_stable_hits + 1 if (lifted and meets) else 0

                    hold_target = ArmTargetPose(
                        pos=retract_pos
                        + np.array(
                            [0.0, 0.0, remaining_lift],
                            dtype=np.float64,
                        ),
                        quat_wxyz=lift_upright_quat.copy(),
                    )
                    no_drop_during_hold = True
                    hold_contact_ok = True
                    for _ in range(lift_hold_steps):
                        info = _record_and_step(close_pose, hold_target, PHASE_LIFT_HOLD)
                        object_z = float(data.xpos[mustard_cfg.body_id][2])
                        object_z_max = max(object_z_max, object_z)
                        lifted = (object_z - object_z_ref) >= args.lift_success_delta
                        if args.task == TASK_PINCH_AND_LIFT:
                            meets = _pinch_contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        else:
                            meets = _contact_meets(
                                info["n_contacts"], info["force"], info["touched"], args
                            )
                        if not lifted:
                            no_drop_during_hold = False
                        if not meets:
                            hold_contact_ok = False
                    if no_drop_during_hold:
                        task_success = True
                        first_success_step = step_counter

                extra_metrics = {
                    "object_z_ref": float(object_z_ref),
                    "object_z_max": float(object_z_max),
                    "object_dz_max": float(object_z_max - object_z_ref),
                    "no_drop_during_hold": bool(no_drop_during_hold),
                    "hold_contact_ok": bool(hold_contact_ok),
                    "lift_hold_steps": int(lift_hold_steps),
                    "lift_target_mode": "latched_palm_retract_then_upright_z_up",
                    "post_grasp_upright_steps": int(args.post_grasp_upright_steps),
                    "post_grasp_retract_steps": int(args.post_grasp_retract_steps),
                    "post_grasp_retract_height": float(args.post_grasp_retract_height),
                    "lift_upright_rot_fraction": float(args.lift_upright_rot_fraction),
                    "lift_upright_rot_power": float(args.lift_upright_rot_power),
                }

            elif args.task == TASK_PUSH_OVER:
                contact_pose = _interpolate_hand_pose(
                    grasp_value=1.0,
                    hand_cfg=hand_cfg,
                    hand_trajectory=spec.hand_trajectory,
                    preshape_pivot=preshape_ratio,
                )
                initial_xy = object_xy_ref.copy()
                topple_angle_max = 0.0
                push_contact_seen = False
                release_after_topple = False
                release_after_topple_hits = 0
                push_planar_disp_max = 0.0
                last_push_target = None
                for i in range(spec.interaction_steps):
                    alpha = (i + 1) / max(spec.interaction_steps, 1)
                    approach_target, interact_target = _resolve_arm_targets(
                        data=data,
                        mustard_cfg=mustard_cfg,
                        approach_offset=approach_offset,
                        push_offset=interact_offset,
                        approach_rot_quat=approach_rot_quat,
                        push_rot_quat=interact_rot_quat,
                        offset_frame="object",
                        rot_frame="object",
                    )
                    push_target = ArmTargetPose(
                        pos=((1.0 - alpha) * approach_target.pos + alpha * interact_target.pos).astype(
                            np.float64
                        ),
                        quat_wxyz=interact_target.quat_wxyz.copy(),
                    )
                    last_push_target = push_target
                    info = _record_and_step(contact_pose, push_target, PHASE_PUSH)
                    object_z = float(info["object_pos"][2])
                    object_z_max = max(object_z_max, object_z)
                    push_planar_disp = float(np.linalg.norm(info["object_pos"][:2] - initial_xy))
                    push_planar_disp_max = max(push_planar_disp_max, push_planar_disp)
                    tilt_now = _tilt_deg(info["object_rot"])
                    topple_angle_max = max(topple_angle_max, tilt_now)
                    toppled = tilt_now >= args.push_success_tilt_deg
                    not_lifted = (object_z - object_z_ref) <= args.push_max_lift_dz
                    pointing_contact = _push_contact_meets(
                        info["n_contacts"], info["force"], info["touched"], args
                    )
                    push_contact_seen = push_contact_seen or pointing_contact
                    released = int(info["n_contacts"]) == 0
                    if toppled and not_lifted:
                        release_after_topple_hits = release_after_topple_hits + 1 if released else 0
                        release_after_topple = (
                            release_after_topple
                            or release_after_topple_hits >= max(int(args.push_release_steps), 1)
                        )
                    else:
                        release_after_topple_hits = 0
                if last_push_target is None:
                    _, last_push_target = _resolve_arm_targets(
                        data=data,
                        mustard_cfg=mustard_cfg,
                        approach_offset=approach_offset,
                        push_offset=interact_offset,
                        approach_rot_quat=approach_rot_quat,
                        push_rot_quat=interact_rot_quat,
                        offset_frame="object",
                        rot_frame="object",
                    )
                for _ in range(spec.final_steps):
                    info = _record_and_step(contact_pose, last_push_target, PHASE_PUSH)
                    object_z = float(info["object_pos"][2])
                    object_z_max = max(object_z_max, object_z)
                    push_planar_disp = float(np.linalg.norm(info["object_pos"][:2] - initial_xy))
                    push_planar_disp_max = max(push_planar_disp_max, push_planar_disp)
                    tilt_now = _tilt_deg(info["object_rot"])
                    topple_angle_max = max(topple_angle_max, tilt_now)
                    toppled = tilt_now >= args.push_success_tilt_deg
                    not_lifted = (object_z - object_z_ref) <= args.push_max_lift_dz
                    pointing_contact = _push_contact_meets(
                        info["n_contacts"], info["force"], info["touched"], args
                    )
                    push_contact_seen = push_contact_seen or pointing_contact
                    released = int(info["n_contacts"]) == 0
                    if toppled and not_lifted:
                        release_after_topple_hits = release_after_topple_hits + 1 if released else 0
                        release_after_topple = (
                            release_after_topple
                            or release_after_topple_hits >= max(int(args.push_release_steps), 1)
                        )
                    else:
                        release_after_topple_hits = 0
                task_success = (
                    topple_angle_max >= args.push_success_tilt_deg
                    and (object_z_max - object_z_ref) <= args.push_max_lift_dz
                    and release_after_topple
                )
                if task_success:
                    first_success_step = step_counter
                extra_metrics = {
                    "tilt_deg_max": float(topple_angle_max),
                    "push_contact_seen": bool(push_contact_seen),
                    "release_after_topple": bool(release_after_topple),
                    "push_planar_disp_max": float(push_planar_disp_max),
                    "object_xy_ref": [float(initial_xy[0]), float(initial_xy[1])],
                    "object_xy_final": [float(data.xpos[mustard_cfg.body_id][0]), float(data.xpos[mustard_cfg.body_id][1])],
                    "object_z_ref": float(object_z_ref),
                    "object_z_max": float(object_z_max),
                    "object_dz_max": float(object_z_max - object_z_ref),
                }

            elif args.task == TASK_HOOK_AND_PULL:
                hook_pose = _interpolate_hand_pose(
                    grasp_value=1.0,
                    hand_cfg=hand_cfg,
                    hand_trajectory=spec.hand_trajectory,
                    preshape_pivot=preshape_ratio,
                )
                initial_x = float(object_xy_ref[0])
                pull_dx_max = 0.0
                hook_contact_seen = False
                approach_target, interact_target = _resolve_arm_targets(
                    data=data,
                    mustard_cfg=mustard_cfg,
                    approach_offset=approach_offset,
                    push_offset=interact_offset,
                    approach_rot_quat=approach_rot_quat,
                    push_rot_quat=interact_rot_quat,
                    offset_frame="object",
                    rot_frame="object",
                )
                world_forward_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                over_target = ArmTargetPose(
                    pos=(interact_target.pos + np.array([0.0, 0.0, 0.030], dtype=np.float64)).astype(
                        np.float64
                    ),
                    quat_wxyz=interact_target.quat_wxyz.copy(),
                )
                sweep_target = ArmTargetPose(
                    pos=(over_target.pos + 0.036 * world_forward_axis).astype(np.float64),
                    quat_wxyz=interact_target.quat_wxyz.copy(),
                )
                engage_target = ArmTargetPose(
                    pos=(sweep_target.pos + np.array([0.0, 0.0, -0.072], dtype=np.float64)).astype(
                        np.float64
                    ),
                    quat_wxyz=interact_target.quat_wxyz.copy(),
                )
                for i in range(spec.interaction_steps):
                    alpha = (i + 1) / max(spec.interaction_steps, 1)
                    if alpha <= 0.30:
                        beta = alpha / 0.30
                        stage_start = approach_target.pos
                        stage_end = over_target.pos
                    elif alpha <= 0.78:
                        beta = (alpha - 0.30) / 0.48
                        stage_start = over_target.pos
                        stage_end = sweep_target.pos
                    else:
                        beta = (alpha - 0.78) / 0.22
                        stage_start = sweep_target.pos
                        stage_end = engage_target.pos
                    hook_target = ArmTargetPose(
                        pos=((1.0 - beta) * stage_start + beta * stage_end).astype(np.float64),
                        quat_wxyz=interact_target.quat_wxyz.copy(),
                    )
                    info = _record_and_step(hook_pose, hook_target, PHASE_CLOSE)
                    object_z = float(info["object_pos"][2])
                    object_z_max = max(object_z_max, object_z)
                    meets = _pull_contact_meets(
                        info["n_contacts"], info["force"], info["touched"], args
                    )
                    hook_contact_seen = hook_contact_seen or meets
                    grasp_stable_hits = grasp_stable_hits + 1 if meets else 0
                # Pull in a longer straight line after the hook engages.
                # Keep y/z fixed to the engage pose so the hand does not drift diagonally.
                # Stop the backward pull earlier; going farther makes the Franka
                # fold into its own body and twist upward near the end.
                retract_distance = 0.200
                retract_target = ArmTargetPose(
                    pos=np.asarray(
                        [
                            engage_target.pos[0] - retract_distance,
                            engage_target.pos[1],
                            engage_target.pos[2],
                        ],
                        dtype=np.float64,
                    ),
                    quat_wxyz=interact_target.quat_wxyz.copy(),
                )
                for i in range(spec.final_steps):
                    alpha = (i + 1) / max(spec.final_steps, 1)
                    pull_target = ArmTargetPose(
                        pos=np.asarray(
                            [
                                (1.0 - alpha) * engage_target.pos[0] + alpha * retract_target.pos[0],
                                engage_target.pos[1],
                                engage_target.pos[2],
                            ],
                            dtype=np.float64,
                        ),
                        quat_wxyz=interact_target.quat_wxyz.copy(),
                    )
                    info = _record_and_step(hook_pose, pull_target, PHASE_PULL)
                    object_z = float(info["object_pos"][2])
                    object_z_max = max(object_z_max, object_z)
                    pull_dx = max(0.0, initial_x - float(info["object_pos"][0]))
                    pull_dx_max = max(pull_dx_max, pull_dx)
                    not_lifted = (object_z - object_z_ref) <= args.pull_max_lift_dz
                    pull_contact = _pull_contact_meets(
                        info["n_contacts"], info["force"], info["touched"], args
                    )
                    hook_contact_seen = hook_contact_seen or pull_contact
                task_success = (
                    hook_contact_seen
                    and pull_dx_max >= args.pull_success_dx
                    and (object_z_max - object_z_ref) <= args.pull_max_lift_dz
                )
                if task_success:
                    first_success_step = step_counter
                extra_metrics = {
                    "object_x_ref": float(initial_x),
                    "object_x_final": float(data.xpos[mustard_cfg.body_id][0]),
                    "pull_dx_max": float(pull_dx_max),
                    "hook_contact_seen": bool(hook_contact_seen),
                    "object_z_ref": float(object_z_ref),
                    "object_z_max": float(object_z_max),
                    "object_dz_max": float(object_z_max - object_z_ref),
                }

            elif args.task == TASK_ROTATE_IN_PLACE:
                support_pose = _interpolate_hand_pose(
                    grasp_value=1.0,
                    hand_cfg=hand_cfg,
                    hand_trajectory=spec.hand_trajectory,
                    preshape_pivot=preshape_ratio,
                )
                yaw_delta_max = 0.0
                xy_drift_max = 0.0
                tilt_deg_max = 0.0
                radius_xy = interact_offset[:2]
                base_angle = float(np.arctan2(radius_xy[1], radius_xy[0]))
                radius = float(np.linalg.norm(radius_xy))
                z_height = float(interact_offset[2])
                for i in range(spec.interaction_steps):
                    alpha = (i + 1) / max(spec.interaction_steps, 1)
                    theta = base_angle + np.deg2rad(args.rotate_arc_deg) * alpha
                    rotate_offset = np.asarray(
                        [radius * np.cos(theta), radius * np.sin(theta), z_height],
                        dtype=np.float32,
                    )
                    _, rotate_target = _resolve_arm_targets(
                        data=data,
                        mustard_cfg=mustard_cfg,
                        approach_offset=approach_offset,
                        push_offset=rotate_offset,
                        approach_rot_quat=approach_rot_quat,
                        push_rot_quat=interact_rot_quat,
                        offset_frame="object",
                        rot_frame="object",
                    )
                    info = _record_and_step(support_pose, rotate_target, PHASE_ROTATE)
                    object_z = float(info["object_pos"][2])
                    object_z_max = max(object_z_max, object_z)
                    tilt_now = _tilt_deg(info["object_rot"])
                    yaw_now = _yaw_rad(info["object_rot"])
                    yaw_delta = _angle_diff_deg(yaw_now, initial_obj_yaw)
                    xy_drift = float(np.linalg.norm(info["object_pos"][:2] - object_xy_ref))
                    tilt_deg_max = max(tilt_deg_max, tilt_now)
                    yaw_delta_max = max(yaw_delta_max, yaw_delta)
                    xy_drift_max = max(xy_drift_max, xy_drift)
                    valid = (
                        yaw_delta >= args.rotate_success_yaw_deg
                        and tilt_now <= args.rotate_max_tilt_deg
                        and xy_drift <= args.rotate_max_xy_drift
                        and (object_z - object_z_ref) <= args.push_max_lift_dz
                    )
                    task_success_stable_hits = task_success_stable_hits + 1 if valid else 0
                for _ in range(spec.final_steps):
                    _record_and_step(idle_pose, None, PHASE_SETTLE)
                task_success = task_success_stable_hits >= args.stable_steps
                if task_success:
                    first_success_step = step_counter
                extra_metrics = {
                    "yaw_delta_deg_max": float(yaw_delta_max),
                    "tilt_deg_max": float(tilt_deg_max),
                    "xy_drift_max": float(xy_drift_max),
                    "object_z_ref": float(object_z_ref),
                    "object_z_max": float(object_z_max),
                    "object_dz_max": float(object_z_max - object_z_ref),
                }
            else:  # pragma: no cover
                raise AssertionError(f"Unhandled task: {args.task}")

            reached = bool(approach_min_err <= args.arm_reach_threshold)
            success = bool(reached and task_success)

            metrics = {
                "task_name": args.task,
                "approach_min_err": float(approach_min_err),
                "approach_min_rot_err_deg": float(approach_min_rot_err),
                "approach_steps_used": int(approach_steps_used),
                "approach_steps_budget": int(spec.approach_steps),
                "reached": reached,
                "grasp_acquired": bool(grasp_acquired),
                "task_success": bool(task_success),
                "first_grasp_step": int(first_grasp_step),
                "first_success_step": int(first_success_step),
                "best_contacts": int(best_contacts),
                "best_force": float(best_force),
                "best_fingers": sorted(best_fingers),
                "steps": int(step_counter),
                **extra_metrics,
            }
            last_attempt_metrics = metrics

            if success:
                ep_name = f"episode_{saved_success:05d}.npz"
                _save_episode_npz(
                    episode_path=raw_dir / ep_name,
                    images=images,
                    states=states,
                    actions=actions,
                    phases=phases,
                    contacts=contacts,
                    arm_cmd_pose_wxyz=arm_cmd_pose_wxyz,
                    arm_obs_pose_wxyz=arm_obs_pose_wxyz,
                    arm_pose_error=arm_pose_error,
                    success=success,
                    instruction=instruction,
                    task_name=args.task,
                    intent=spec.intent,
                    side=args.side,
                    control_hz=float(effective_control_hz),
                    capture_hz=float(effective_capture_hz),
                    object_qpos=object_qpos,
                    criteria=criteria,
                    metrics=metrics,
                )
                saved_success += 1
                status = "SAVE"
            else:
                status = "SKIP"

            print(
                f"[attempt {attempts:04d}] {status} task={args.task} "
                f"saved={saved_success}/{args.target_episodes} "
                f"reach={int(reached)} success={int(success)} "
                f"err={approach_min_err:.4f} rot={approach_min_rot_err:.2f}deg "
                f"contacts={best_contacts} fingers={','.join(sorted(best_fingers)) or '-'}"
            )

        if viewer_ctx is not None and args.keep_open:
            print("Viewer is kept open. Close the MuJoCo window to finish.")
            while viewer_ctx.is_running():
                viewer_ctx.sync()
                if viewer_delay > 0.0:
                    time.sleep(viewer_delay)
    finally:
        renderer.close()
        if record_renderer is not None:
            record_renderer.close()
        if writer is not None:
            writer.close()
            print(f"[record] finished {record_path}", flush=True)
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception as exc:  # pragma: no cover
                print(f"[warn] viewer cleanup error: {exc}")

    summary = {
        "finished": saved_success >= args.target_episodes,
        "saved_success_episodes": int(saved_success),
        "target_episodes": int(args.target_episodes),
        "attempts": int(attempts),
        "max_attempts": int(args.max_attempts),
        "out_dir": str(out_dir),
        "raw_dir": str(raw_dir),
        "task_name": args.task,
        "instruction": instruction,
        "intent": spec.intent.__dict__,
        "side": args.side,
        "control_hz": float(effective_control_hz),
        "capture_hz": float(effective_capture_hz),
        "capture_every": int(capture_every),
        "action_semantics": "joint23_absolute_target_object6d_ik",
        "image_shape": [args.image_height, args.image_width, 3],
        "state_dim": 7 + 16 + 7 + 16 + 3 + 3 + 4 + 4,
        "action_dim": 23,
        "criteria": criteria,
        "last_attempt_metrics": last_attempt_metrics,
        "policy": {
            "hand_trajectory": spec.hand_trajectory,
            "preshape_ratio": float(preshape_ratio),
            "spawn_offset_local": [float(x) for x in spec.spawn_offset_local],
            "approach_offset": [float(x) for x in spec.approach_offset],
            "interact_offset": [float(x) for x in spec.interact_offset],
            "approach_rot_quat_wxyz": [float(x) for x in spec.approach_rot_quat],
            "interact_rot_quat_wxyz": [float(x) for x in spec.interact_rot_quat],
            "settle_steps": int(settle_steps),
            "approach_steps": int(spec.approach_steps),
            "approach_early_exit_pos_err": float(args.approach_early_exit_pos_err),
            "approach_early_exit_rot_err_deg": float(args.approach_early_exit_rot_err_deg),
            "approach_early_exit_stable_steps": int(args.approach_early_exit_stable_steps),
            "preshape_steps": int(spec.preshape_steps),
            "interaction_steps": int(spec.interaction_steps),
            "close_hold_steps": int(spec.close_hold_steps),
            "final_steps": int(spec.final_steps),
            "lock_object_until_interaction": bool(spec.lock_object_until_interaction),
            "arm_disturb_xyz": [float(x) for x in arm_disturb_xyz],
            "arm_disturb_resample_steps": int(arm_disturb_resample_steps),
            "arm_disturb_phases": sorted(disturb_phase_tokens),
            "seed": int(args.seed),
        },
    }
    summary_path = out_dir / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
