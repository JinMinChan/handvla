#!/usr/bin/env python3
"""Run closed-loop Octo rollout on the mustard intent benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from dataclasses import dataclass, field
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
from scripts.data.collect_mustard_intent_benchmark import (
    TASK_HOOK_AND_PULL,
    TASK_PUSH_OVER,
    TASK_WRAP_AND_LIFT,
    _angle_diff_deg,
    _build_hand_config,
    _build_mustard_config,
    _build_contact_config,
    _build_arm_config,
    _contact_meets,
    _detect_contact_with_target,
    _make_state_vector,
    _normalize_quat,
    _pull_contact_meets,
    _push_contact_meets,
    _sample_spawn_pose,
    _set_mustard_pose,
    _task_spec,
    _tilt_deg,
    _yaw_rad,
)
from scripts.data.collect_pickandlift_rlds import (
    ArmTargetPose,
    _capture_every,
    _capture_frame,
    _quat_to_rot,
    _rot_to_quat,
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


TASKS = (TASK_WRAP_AND_LIFT, TASK_PUSH_OVER, TASK_HOOK_AND_PULL)


@dataclass
class EpisodeMetrics:
    task_name: str
    episode_idx: int
    success: bool = False
    reached: bool = False
    nan_action: bool = False
    action_dim_error: bool = False
    best_contacts: int = 0
    best_force: float = 0.0
    best_fingers: set[str] = field(default_factory=set)
    approach_min_err: float = 1e9
    approach_min_rot_err_deg: float = 1e9
    object_z_ref: float = 0.0
    object_z_max: float = -1e9
    topple_angle_max: float = 0.0
    release_after_topple: bool = False
    push_planar_disp_max: float = 0.0
    push_contact_seen: bool = False
    hook_contact_seen: bool = False
    pull_dx_max: float = 0.0
    wrap_hold_hits: int = 0
    wrap_latched_hold: bool = False
    steps: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate finetuned Octo model on mustard intent benchmark tasks."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to finetuned Octo checkpoint directory.",
    )
    parser.add_argument(
        "--basis-path",
        type=str,
        required=True,
        help="Path to mustard intent PCA basis npz.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mustard_grasp_oxe",
        help="Key inside model.dataset_statistics for action/proprio normalization.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="wrap_and_lift,push_over,hook_and_pull",
        help="Comma-separated tasks to evaluate.",
    )
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--max-policy-steps", type=int, default=100)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--capture-hz", type=float, default=5.0)
    parser.add_argument("--policy-repeat", type=int, default=20)
    parser.add_argument(
        "--execute-actions-per-plan",
        type=int,
        default=1,
        help=(
            "How many predicted dataset-rate actions to execute from each sampled action chunk "
            "before replanning. For action_horizon>1, values like 2 implement receding-horizon execution."
        ),
    )
    parser.add_argument("--action-smoothing", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--policy-image-size", type=int, default=256)

    parser.add_argument("--spawn-pos", type=float, nargs=3, default=(0.78, 0.12, 0.82))
    parser.add_argument("--spawn-quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--spawn-jitter-xy", type=float, default=0.005)
    parser.add_argument("--spawn-yaw-jitter-deg", type=float, default=3.0)

    parser.add_argument("--ik-gain", type=float, default=0.95)
    parser.add_argument("--ik-rot-gain", type=float, default=0.9)
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument("--ik-rot-weight", type=float, default=0.20)
    parser.add_argument("--ik-max-joint-step", type=float, default=0.08)

    parser.add_argument("--arm-reach-threshold", type=float, default=0.05)
    parser.add_argument("--lift-success-delta", type=float, default=0.08)
    parser.add_argument("--lift-hold-seconds", type=float, default=1.5)
    parser.add_argument(
        "--wrap-hold-latch-dz",
        type=float,
        default=-1.0,
        help=(
            "If >= 0, once wrap contact is valid and object dz exceeds this value, "
            "freeze the current arm/hand commands for the rest of the rollout."
        ),
    )
    parser.add_argument(
        "--wrap-hold-latch-mode",
        choices=("freeze", "monotone_z"),
        default="freeze",
        help="How to stabilize the wrap lift after latch.",
    )
    parser.add_argument("--min-contacts", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=3)
    parser.add_argument("--require-thumb-contact", action="store_true", default=True)
    parser.add_argument("--no-require-thumb-contact", dest="require_thumb_contact", action="store_false")
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--max-force", type=float, default=1000.0)
    parser.add_argument("--stable-steps", type=int, default=3)
    parser.add_argument("--close-z-clearance", type=float, default=0.015)

    parser.add_argument("--push-success-tilt-deg", type=float, default=55.0)
    parser.add_argument("--push-max-lift-dz", type=float, default=0.04)
    parser.add_argument("--push-release-steps", type=int, default=3)

    parser.add_argument("--pull-success-dx", type=float, default=0.08)
    parser.add_argument("--pull-max-lift-dz", type=float, default=0.04)

    parser.add_argument("--post-settle-steps", type=int, default=2000)

    parser.add_argument("--viewer", dest="viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.set_defaults(viewer=False)
    parser.add_argument("--viewer-step", type=float, default=60.0)
    parser.add_argument("--viewer-step-delay", type=float, default=None)
    parser.add_argument("--keep-open", action="store_true")

    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-width", type=int, default=1920)
    parser.add_argument("--record-height", type=int, default=1080)
    parser.add_argument("--record-fps", type=int, default=60)
    parser.add_argument(
        "--record-dir",
        type=str,
        default="codex/logs/videos",
        help="Per-task episode videos will be written here when --record is set.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output path for evaluation summary json.",
    )
    return parser.parse_args()


def _default_summary_path() -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return Path("codex/logs/json") / f"mustard_intent_octo_eval_{ts}.json"


def _load_basis(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "mu" not in data or "B" not in data:
        raise KeyError(f"Basis file missing mu/B: {path}")
    mu = np.asarray(data["mu"], dtype=np.float32).reshape(-1)
    B = np.asarray(data["B"], dtype=np.float32)
    if mu.shape[0] != 16:
        raise ValueError(f"mu must be length 16, got {mu.shape}")
    if B.ndim != 2 or B.shape[0] != 16:
        raise ValueError(f"B must be shape (16,k), got {B.shape}")
    return mu, B


def _find_stats(model: OctoModel, dataset_name: str) -> tuple[dict | None, dict | None]:
    stats = model.dataset_statistics
    if not isinstance(stats, dict):
        return None, None
    if "action" in stats:
        return stats.get("action"), stats.get("proprio")
    if dataset_name in stats:
        ds = stats[dataset_name]
        if isinstance(ds, dict):
            return ds.get("action"), ds.get("proprio")
    first_key = next(iter(stats.keys()), None)
    if first_key is None:
        return None, None
    ds = stats[first_key]
    if isinstance(ds, dict):
        return ds.get("action"), ds.get("proprio")
    return None, None


def _euler_xyz_to_quat(euler_xyz: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in np.asarray(euler_xyz, dtype=np.float64)]
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    quat = np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    return _normalize_quat(quat).astype(np.float32)


def _resolve_target(
    data: mujoco.MjData,
    mustard_cfg,
    offset: np.ndarray,
    rot_quat: np.ndarray,
) -> ArmTargetPose:
    mustard_pos = data.xpos[mustard_cfg.body_id].copy()
    obj_rot = data.xmat[mustard_cfg.body_id].reshape(3, 3)
    target_pos = mustard_pos + obj_rot @ offset
    target_rot = obj_rot @ _quat_to_rot(rot_quat)
    return ArmTargetPose(pos=target_pos.astype(np.float64), quat_wxyz=_rot_to_quat(target_rot))


def _init_writer(path: Path, fps: int) -> imageio.Writer:
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
    )


def run_eval(args: argparse.Namespace) -> dict:
    tf.config.set_visible_devices([], "GPU")
    rng = np.random.default_rng(args.seed)
    jax_key = jax.random.PRNGKey(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        if task not in TASKS:
            raise SystemExit(f"Unsupported task: {task}")

    basis_path = Path(args.basis_path).expanduser().resolve()
    mu, B = _load_basis(basis_path)
    k = int(B.shape[1])
    expected_action_dim = 6 + k

    mjcf = franka_allegro_mjcf.load(side=args.side, add_mustard=True)
    model = mjcf.compile()
    data = mujoco.MjData(model)

    control_target_dt = 1.0 / max(float(args.control_hz), 1e-6)
    control_nstep = max(1, int(round(control_target_dt / model.opt.timestep)))
    effective_control_dt = control_nstep * model.opt.timestep
    effective_control_hz = 1.0 / max(effective_control_dt, 1e-9)
    capture_every = _capture_every(effective_control_hz, args.record_fps if args.record else args.capture_hz)
    lift_hold_steps = max(1, int(np.ceil(args.lift_hold_seconds * effective_control_hz)))

    arm_cfg = _build_arm_config(model, args.side)
    hand_cfg = _build_hand_config(model, args.side)
    mustard_cfg = _build_mustard_config(model)
    contact_cfg = _build_contact_config(model, args.side, mustard_cfg.body_id)
    force_buf = np.zeros(6, dtype=float)

    policy_model = OctoModel.load_pretrained(args.model_path)
    if policy_model.text_processor is None:
        raise RuntimeError("Loaded model has no text_processor.")
    model_window_size = int(policy_model.example_batch["observation"]["timestep_pad_mask"].shape[1])
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
    policy_cam = mujoco.MjvCamera()
    set_default_franka_allegro_camera(policy_cam)
    record_renderer = None
    if args.record:
        record_renderer = mujoco.Renderer(model, width=args.record_width, height=args.record_height)

    base_spawn_pos = np.asarray(args.spawn_pos, dtype=np.float32)
    base_spawn_quat = np.asarray(args.spawn_quat, dtype=np.float32)
    policy_repeat = max(1, int(args.policy_repeat))
    alpha = float(np.clip(args.action_smoothing, 0.0, 1.0))

    summary = {
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "basis_path": str(basis_path),
        "dataset_name": args.dataset_name,
        "action_dim": int(expected_action_dim),
        "window_size": int(model_window_size),
        "tasks": {},
        "episodes": [],
    }

    try:
        for task_name in tasks:
            spec = _task_spec(task_name)
            task = {"language_instruction": policy_model.text_processor.encode([spec.instruction])}
            task_successes = 0
            task_eps: list[dict] = []

            for ep in range(args.episodes_per_task):
                initial_state = model.key("initial_state").id
                mujoco.mj_resetDataKeyframe(model, data, initial_state)
                mujoco.mj_forward(model, data)

                spawn_pos, spawn_quat = _sample_spawn_pose(base_spawn_pos, base_spawn_quat, rng, args)
                _set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_forward(model, data)

                q_arm_cmd = data.qpos[arm_cfg.qpos_ids].astype(np.float32).copy()
                q_arm_cmd = np.clip(q_arm_cmd, arm_cfg.q_min, arm_cfg.q_max)
                q_hand_cmd = data.qpos[hand_cfg.qpos_ids].astype(np.float32).copy()
                q_hand_cmd = np.clip(q_hand_cmd, hand_cfg.q_min, hand_cfg.q_max)
                cached_arm_target = ArmTargetPose(
                    pos=data.xpos[arm_cfg.palm_body_id].astype(np.float64).copy(),
                    quat_wxyz=_rot_to_quat(
                        data.xmat[arm_cfg.palm_body_id].reshape(3, 3).copy()
                    ).astype(np.float64),
                )
                cached_hand_target = q_hand_cmd.copy()
                policy_image_history: list[np.ndarray] = []
                policy_state_history: list[np.ndarray] = []

                metrics = EpisodeMetrics(task_name=task_name, episode_idx=ep)
                metrics.object_z_ref = float(data.xpos[mustard_cfg.body_id][2])
                initial_xy = data.xpos[mustard_cfg.body_id][:2].copy()
                initial_x = float(initial_xy[0])
                initial_yaw = _yaw_rad(data.xmat[mustard_cfg.body_id].reshape(3, 3))
                writer = None
                if args.record and ep == 0 and record_renderer is not None:
                    video_name = f"mustard_intent_octo_{task_name}_demo_{datetime.now().strftime('%y%m%d_%H%M%S')}.mp4"
                    writer = _init_writer(Path(args.record_dir) / video_name, args.record_fps)

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
                            [[False] * pad_len + [True] * valid_len], dtype=np.bool_
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

                action_queue: list[np.ndarray] = []
                wrap_latched_hold = False
                wrap_latched_arm_target: ArmTargetPose | None = None
                wrap_latched_hand_target: np.ndarray | None = None
                for p_step in range(args.max_policy_steps):
                    metrics.steps += 1

                    if not action_queue:
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
                            pred_chunk = policy_model.sample_actions(
                                obs, task, unnormalization_statistics=action_stats, rng=key
                            )[0]
                        else:
                            pred_chunk = policy_model.sample_actions(obs, task, rng=key)[0]
                        pred_chunk = np.asarray(pred_chunk, dtype=np.float32)
                        if pred_chunk.ndim == 1:
                            pred_chunk = pred_chunk[None, :]
                        if pred_chunk.ndim != 2 or pred_chunk.shape[-1] != expected_action_dim:
                            metrics.action_dim_error = True
                            break
                        if not np.all(np.isfinite(pred_chunk)):
                            metrics.nan_action = True
                            break
                        execute_n = max(
                            1,
                            min(int(args.execute_actions_per_plan), int(pred_chunk.shape[0])),
                        )
                        action_queue = [pred_chunk[i].copy() for i in range(execute_n)]

                    pred = action_queue.pop(0)
                    if not wrap_latched_hold:
                        arm_pred = pred[:6]
                        hand_pred = pred[6:]
                        cached_arm_target = ArmTargetPose(
                            pos=np.asarray(arm_pred[:3], dtype=np.float64).copy(),
                            quat_wxyz=_euler_xyz_to_quat(arm_pred[3:6]).astype(np.float64),
                        )
                        cached_hand_target = mu + B @ hand_pred
                        cached_hand_target = np.clip(cached_hand_target, hand_cfg.q_min, hand_cfg.q_max)

                        q_arm_des, arm_err, arm_rot_err = _step_arm_ik(
                            model=model,
                            data=data,
                            arm_cfg=arm_cfg,
                            target=cached_arm_target,
                            gain=float(args.ik_gain),
                            rot_gain=float(args.ik_rot_gain),
                            damping=float(args.ik_damping),
                            rot_weight=float(args.ik_rot_weight),
                            max_joint_step=float(args.ik_max_joint_step),
                        )
                        q_arm_cmd = (1.0 - alpha) * q_arm_cmd + alpha * q_arm_des
                        q_hand_cmd = (1.0 - alpha) * q_hand_cmd + alpha * cached_hand_target
                        q_arm_cmd = np.clip(q_arm_cmd, arm_cfg.q_min, arm_cfg.q_max)
                        q_hand_cmd = np.clip(q_hand_cmd, hand_cfg.q_min, hand_cfg.q_max)
                    elif args.wrap_hold_latch_mode == "monotone_z" and wrap_latched_arm_target is not None:
                        arm_pred = pred[:6]
                        candidate_target = ArmTargetPose(
                            pos=np.asarray(arm_pred[:3], dtype=np.float64).copy(),
                            quat_wxyz=_euler_xyz_to_quat(arm_pred[3:6]).astype(np.float64),
                        )
                        wrap_latched_arm_target.pos[2] = max(
                            float(wrap_latched_arm_target.pos[2]),
                            float(candidate_target.pos[2]),
                        )
                        cached_arm_target = ArmTargetPose(
                            pos=wrap_latched_arm_target.pos.copy(),
                            quat_wxyz=wrap_latched_arm_target.quat_wxyz.copy(),
                        )
                        if wrap_latched_hand_target is not None:
                            cached_hand_target = wrap_latched_hand_target.copy()
                        q_arm_des, arm_err, arm_rot_err = _step_arm_ik(
                            model=model,
                            data=data,
                            arm_cfg=arm_cfg,
                            target=cached_arm_target,
                            gain=float(args.ik_gain),
                            rot_gain=float(args.ik_rot_gain),
                            damping=float(args.ik_damping),
                            rot_weight=float(args.ik_rot_weight),
                            max_joint_step=float(args.ik_max_joint_step),
                        )
                        q_arm_cmd = (1.0 - alpha) * q_arm_cmd + alpha * q_arm_des
                        q_arm_cmd = np.clip(q_arm_cmd, arm_cfg.q_min, arm_cfg.q_max)
                        q_hand_cmd = np.clip(cached_hand_target, hand_cfg.q_min, hand_cfg.q_max)

                    for _ in range(policy_repeat):
                        data.ctrl[arm_cfg.act_ids] = q_arm_cmd
                        data.ctrl[7:23] = q_hand_cmd
                        mujoco.mj_step(model, data, nstep=control_nstep)

                        n_post, f_post, touched_post = _detect_contact_with_target(
                            model, data, contact_cfg, force_buf
                        )
                        metrics.best_contacts = max(metrics.best_contacts, int(n_post))
                        metrics.best_force = max(metrics.best_force, float(f_post))
                        if len(touched_post) >= len(metrics.best_fingers):
                            metrics.best_fingers = set(touched_post)

                        metrics.object_z_max = max(metrics.object_z_max, float(data.xpos[mustard_cfg.body_id][2]))

                        object_pos = data.xpos[mustard_cfg.body_id].copy()
                        object_rot = data.xmat[mustard_cfg.body_id].reshape(3, 3).copy()
                        current_target = _resolve_target(
                            data,
                            mustard_cfg,
                            np.asarray(spec.interact_offset, dtype=np.float32),
                            np.asarray(spec.interact_rot_quat, dtype=np.float64),
                        )
                        ee_pos = data.xpos[arm_cfg.palm_body_id].copy()
                        metrics.approach_min_err = min(
                            metrics.approach_min_err, float(np.linalg.norm(current_target.pos - ee_pos))
                        )
                        ee_rot = data.xmat[arm_cfg.palm_body_id].reshape(3, 3).copy()
                        current_tgt_rot = _quat_to_rot(current_target.quat_wxyz)
                        current_rot_err = 0.5 * (
                            np.cross(ee_rot[:, 0], current_tgt_rot[:, 0])
                            + np.cross(ee_rot[:, 1], current_tgt_rot[:, 1])
                            + np.cross(ee_rot[:, 2], current_tgt_rot[:, 2])
                        )
                        metrics.approach_min_rot_err_deg = min(
                            metrics.approach_min_rot_err_deg,
                            float(np.rad2deg(np.linalg.norm(current_rot_err))),
                        )
                        if task_name == TASK_WRAP_AND_LIFT:
                            lifted = (float(object_pos[2]) - metrics.object_z_ref) >= args.lift_success_delta
                            meets = _contact_meets(int(n_post), float(f_post), touched_post, args)
                            metrics.wrap_hold_hits = metrics.wrap_hold_hits + 1 if (lifted and meets) else 0
                            if (
                                not wrap_latched_hold
                                and args.wrap_hold_latch_dz >= 0.0
                                and (float(object_pos[2]) - metrics.object_z_ref) >= args.wrap_hold_latch_dz
                                and meets
                            ):
                                wrap_latched_hold = True
                                metrics.wrap_latched_hold = True
                                wrap_latched_arm_target = ArmTargetPose(
                                    pos=cached_arm_target.pos.copy(),
                                    quat_wxyz=cached_arm_target.quat_wxyz.copy(),
                                )
                                wrap_latched_hand_target = q_hand_cmd.copy()
                            if metrics.wrap_hold_hits >= lift_hold_steps:
                                metrics.success = True
                        elif task_name == TASK_PUSH_OVER:
                            tilt_now = _tilt_deg(object_rot)
                            metrics.topple_angle_max = max(metrics.topple_angle_max, tilt_now)
                            push_planar_disp = float(np.linalg.norm(object_pos[:2] - initial_xy))
                            metrics.push_planar_disp_max = max(metrics.push_planar_disp_max, push_planar_disp)
                            metrics.push_contact_seen = metrics.push_contact_seen or _push_contact_meets(
                                int(n_post), float(f_post), touched_post, args
                            )
                        elif task_name == TASK_HOOK_AND_PULL:
                            pull_dx = max(0.0, initial_x - float(object_pos[0]))
                            metrics.pull_dx_max = max(metrics.pull_dx_max, pull_dx)
                            metrics.hook_contact_seen = metrics.hook_contact_seen or _pull_contact_meets(
                                int(n_post), float(f_post), touched_post, args
                            )

                        if writer is not None and record_renderer is not None:
                            cam = viewer_ctx.cam if viewer_ctx is not None else capture_cam
                            record_renderer.update_scene(data, camera=cam)
                            writer.append_data(record_renderer.render())
                        if viewer_ctx is not None:
                            viewer_ctx.sync()
                        if viewer_delay > 0.0:
                            time.sleep(viewer_delay)

                    if metrics.success:
                        break

                # Passive settle for dynamic tasks and final success computation.
                release_after_topple_hits = 0
                for _ in range(args.post_settle_steps):
                    data.ctrl[arm_cfg.act_ids] = q_arm_cmd
                    data.ctrl[7:23] = q_hand_cmd
                    mujoco.mj_step(model, data, nstep=control_nstep)

                    n_post, f_post, touched_post = _detect_contact_with_target(
                        model, data, contact_cfg, force_buf
                    )
                    metrics.best_contacts = max(metrics.best_contacts, int(n_post))
                    metrics.best_force = max(metrics.best_force, float(f_post))
                    if len(touched_post) >= len(metrics.best_fingers):
                        metrics.best_fingers = set(touched_post)

                    object_pos = data.xpos[mustard_cfg.body_id].copy()
                    object_rot = data.xmat[mustard_cfg.body_id].reshape(3, 3).copy()
                    metrics.object_z_max = max(metrics.object_z_max, float(object_pos[2]))
                    if task_name == TASK_WRAP_AND_LIFT:
                        lifted = (float(object_pos[2]) - metrics.object_z_ref) >= args.lift_success_delta
                        meets = _contact_meets(int(n_post), float(f_post), touched_post, args)
                        metrics.wrap_hold_hits = metrics.wrap_hold_hits + 1 if (lifted and meets) else 0
                        if metrics.wrap_hold_hits >= lift_hold_steps:
                            metrics.success = True
                    elif task_name == TASK_PUSH_OVER:
                        tilt_now = _tilt_deg(object_rot)
                        metrics.topple_angle_max = max(metrics.topple_angle_max, tilt_now)
                        toppled = tilt_now >= args.push_success_tilt_deg
                        not_lifted = (metrics.object_z_max - metrics.object_z_ref) <= args.push_max_lift_dz
                        released = int(n_post) == 0
                        if toppled and not_lifted:
                            release_after_topple_hits = release_after_topple_hits + 1 if released else 0
                            metrics.release_after_topple = (
                                metrics.release_after_topple
                                or release_after_topple_hits >= max(int(args.push_release_steps), 1)
                            )
                        else:
                            release_after_topple_hits = 0
                    elif task_name == TASK_HOOK_AND_PULL:
                        pull_dx = max(0.0, initial_x - float(object_pos[0]))
                        metrics.pull_dx_max = max(metrics.pull_dx_max, pull_dx)
                        metrics.hook_contact_seen = metrics.hook_contact_seen or _pull_contact_meets(
                            int(n_post), float(f_post), touched_post, args
                        )

                    if writer is not None and record_renderer is not None:
                        cam = viewer_ctx.cam if viewer_ctx is not None else capture_cam
                        record_renderer.update_scene(data, camera=cam)
                        writer.append_data(record_renderer.render())

                if task_name == TASK_PUSH_OVER:
                    metrics.success = (
                        metrics.topple_angle_max >= args.push_success_tilt_deg
                        and (metrics.object_z_max - metrics.object_z_ref) <= args.push_max_lift_dz
                        and metrics.release_after_topple
                    )
                elif task_name == TASK_HOOK_AND_PULL:
                    metrics.success = (
                        metrics.hook_contact_seen
                        and metrics.pull_dx_max >= args.pull_success_dx
                        and (metrics.object_z_max - metrics.object_z_ref) <= args.pull_max_lift_dz
                    )

                interact_target = _resolve_target(
                    data,
                    mustard_cfg,
                    np.asarray(spec.interact_offset, dtype=np.float32),
                    np.asarray(spec.interact_rot_quat, dtype=np.float64),
                )
                final_ee_pos = data.xpos[arm_cfg.palm_body_id].copy()
                metrics.approach_min_err = min(
                    metrics.approach_min_err, float(np.linalg.norm(interact_target.pos - final_ee_pos))
                )
                final_ee_rot = data.xmat[arm_cfg.palm_body_id].reshape(3, 3).copy()
                tgt_rot = _quat_to_rot(interact_target.quat_wxyz)
                rot_err = 0.5 * (
                    np.cross(final_ee_rot[:, 0], tgt_rot[:, 0])
                    + np.cross(final_ee_rot[:, 1], tgt_rot[:, 1])
                    + np.cross(final_ee_rot[:, 2], tgt_rot[:, 2])
                )
                metrics.approach_min_rot_err_deg = min(
                    metrics.approach_min_rot_err_deg, float(np.rad2deg(np.linalg.norm(rot_err)))
                )
                metrics.reached = metrics.approach_min_err <= args.arm_reach_threshold
                metrics.success = bool(metrics.reached and metrics.success)

                ep_entry = {
                    "task_name": task_name,
                    "episode": int(ep),
                    "success": bool(metrics.success),
                    "reached": bool(metrics.reached),
                    "nan_action": bool(metrics.nan_action),
                    "action_dim_error": bool(metrics.action_dim_error),
                    "approach_min_err": float(metrics.approach_min_err),
                    "approach_min_rot_err_deg": float(metrics.approach_min_rot_err_deg),
                    "best_contacts": int(metrics.best_contacts),
                    "best_force": float(metrics.best_force),
                    "best_fingers": sorted(metrics.best_fingers),
                    "object_dz_max": float(metrics.object_z_max - metrics.object_z_ref),
                    "wrap_hold_hits": int(metrics.wrap_hold_hits),
                    "wrap_latched_hold": bool(metrics.wrap_latched_hold),
                    "tilt_deg_max": float(metrics.topple_angle_max),
                    "release_after_topple": bool(metrics.release_after_topple),
                    "push_contact_seen": bool(metrics.push_contact_seen),
                    "push_planar_disp_max": float(metrics.push_planar_disp_max),
                    "hook_contact_seen": bool(metrics.hook_contact_seen),
                    "pull_dx_max": float(metrics.pull_dx_max),
                    "steps": int(metrics.steps),
                }
                summary["episodes"].append(ep_entry)
                task_eps.append(ep_entry)
                task_successes += int(metrics.success)

                if writer is not None:
                    writer.close()

            summary["tasks"][task_name] = {
                "episodes": len(task_eps),
                "successes": int(task_successes),
                "success_rate": float(task_successes / max(len(task_eps), 1)),
                "episodes_data": task_eps,
            }

        if viewer_ctx is not None and args.keep_open:
            while viewer_ctx.is_running():
                viewer_ctx.sync()
                if viewer_delay > 0.0:
                    time.sleep(viewer_delay)
    finally:
        renderer_obs.close()
        if record_renderer is not None:
            record_renderer.close()
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass

    total_eps = len(summary["episodes"])
    total_success = sum(int(ep["success"]) for ep in summary["episodes"])
    summary["overall"] = {
        "episodes": int(total_eps),
        "successes": int(total_success),
        "success_rate": float(total_success / max(total_eps, 1)),
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = run_eval(args)
    out_path = Path(args.save_json) if args.save_json else _default_summary_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
