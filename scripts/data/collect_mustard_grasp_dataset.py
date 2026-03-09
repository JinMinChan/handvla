#!/usr/bin/env python3
"""Collect successful mustard grasp episodes and save raw NPZ dataset."""

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
from pathlib import Path
import time

import mujoco
import numpy as np
from mujoco import viewer

from scripts.data.collect_mustard_grasp import (
    POWER_GRASP_CLOSE_QPOS,
    POWER_GRASP_PRESHAPE_QPOS,
    build_contact_config,
    build_hand_config,
    build_mustard_config,
    compute_mustard_spawn_pose,
    detect_contact_with_target,
    reset_to_initial,
    set_mustard_pose,
)
from env import allegro_hand_mjcf
from env.viewer_utils import set_default_hand_camera

FINGER_KEYS = ("ff", "mf", "rf", "th")
PHASE_OPEN = 0
PHASE_PRESHAPE = 1
PHASE_CLOSE = 2
PHASE_HOLD = 3
DEFAULT_TCP12_OUT_DIR = "dataset/mustard_grasp"
DEFAULT_FULL_JOINT_OUT_DIR = "dataset/mustard_grasp_full_joint"


@dataclass(frozen=True)
class JointIndexConfig:
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    tcp_site_ids: np.ndarray
    palm_body_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect mustard grasp raw episodes (save only success until target count)."
    )
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--target-episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_TCP12_OUT_DIR,
        help=(
            "Output root directory. Raw episodes are saved in <out-dir>/raw. "
            "If left default and --action-interface joint16, "
            f"it is auto-switched to {DEFAULT_FULL_JOINT_OUT_DIR}."
        ),
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="grasp the mustard bottle",
        help="Episode-level language instruction.",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--capture-hz", type=float, default=20.0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--action-interface",
        choices=("tcp12", "joint16"),
        default="tcp12",
        help="Saved action interface. tcp12=4 fingertip TCP xyz command next-deltas, joint16=raw joint targets.",
    )
    parser.add_argument(
        "--tcp12-frame",
        choices=("palm_local", "world"),
        default="palm_local",
        help="Frame for tcp12 command next-delta when --action-interface tcp12.",
    )

    parser.add_argument("--open-steps", type=int, default=30)
    parser.add_argument("--preshape-steps", type=int, default=25)
    parser.add_argument("--close-steps", type=int, default=100)
    parser.add_argument("--hold-steps", type=int, default=40)
    parser.add_argument(
        "--open-steps-jitter",
        type=int,
        default=0,
        help="Per-episode jitter range for open steps (uniform integer in [-jitter, +jitter]).",
    )
    parser.add_argument(
        "--preshape-steps-jitter",
        type=int,
        default=0,
        help="Per-episode jitter range for preshape steps.",
    )
    parser.add_argument(
        "--close-steps-jitter",
        type=int,
        default=0,
        help="Per-episode jitter range for close steps.",
    )
    parser.add_argument(
        "--hold-steps-jitter",
        type=int,
        default=0,
        help="Per-episode jitter range for hold steps.",
    )
    parser.add_argument(
        "--spawn-jitter-xy",
        type=float,
        default=0.0,
        help="Per-episode uniform spawn jitter in world XY (meters).",
    )
    parser.add_argument(
        "--spawn-jitter-z",
        type=float,
        default=0.0,
        help="Per-episode uniform spawn jitter in world Z (meters).",
    )
    parser.add_argument(
        "--spawn-yaw-jitter-deg",
        type=float,
        default=0.0,
        help="Per-episode uniform mustard yaw jitter around world Z (degrees).",
    )
    parser.add_argument(
        "--preshape-noise-std",
        type=float,
        default=0.0,
        help="Gaussian std (rad) for per-joint preshape target noise.",
    )
    parser.add_argument(
        "--close-noise-std",
        type=float,
        default=0.0,
        help="Gaussian std (rad) for per-joint close target noise.",
    )

    parser.add_argument("--min-contacts", type=int, default=2)
    parser.add_argument("--min-contact-fingers", type=int, default=2)
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
        help="Disable thumb-contact requirement.",
    )
    parser.add_argument("--min-force", type=float, default=0.5)
    parser.add_argument("--max-force", type=float, default=1000.0)
    parser.add_argument("--stable-steps", type=int, default=3)

    parser.add_argument(
        "--viewer",
        dest="viewer",
        action="store_true",
        help="Visualize collection attempts (default: off).",
    )
    parser.add_argument(
        "--no-viewer",
        dest="viewer",
        action="store_false",
        help="Disable viewer.",
    )
    parser.set_defaults(viewer=False)
    parser.add_argument("--viewer-step", type=float, default=60.0)
    parser.add_argument("--viewer-step-delay", type=float, default=None)
    parser.add_argument("--keep-open", action="store_true")

    return parser.parse_args()


def _build_joint_indices(model: mujoco.MjModel, side: str) -> JointIndexConfig:
    prefix = f"allegro_{side}"
    qpos_ids: list[int] = []
    dof_ids: list[int] = []
    tcp_site_ids: list[int] = []

    for finger in FINGER_KEYS:
        for j in range(4):
            joint_name = f"{prefix}/{finger}j{j}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found: {joint_name}")
            qpos_ids.append(int(model.jnt_qposadr[joint_id]))
            dof_ids.append(int(model.jnt_dofadr[joint_id]))

        tcp_name = f"{prefix}/{finger}_tcp"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_name)
        if site_id < 0:
            raise RuntimeError(f"TCP site not found: {tcp_name}")
        tcp_site_ids.append(int(site_id))

    palm_body_name = f"{prefix}/palm"
    palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_body_name)
    if palm_body_id < 0:
        raise RuntimeError(f"Palm body not found: {palm_body_name}")

    return JointIndexConfig(
        qpos_ids=np.asarray(qpos_ids, dtype=int),
        dof_ids=np.asarray(dof_ids, dtype=int),
        tcp_site_ids=np.asarray(tcp_site_ids, dtype=int),
        palm_body_id=int(palm_body_id),
    )


def _capture_frame(renderer: mujoco.Renderer, data: mujoco.MjData) -> np.ndarray:
    renderer.update_scene(data)
    return renderer.render().copy()


def _make_state_vector(
    data: mujoco.MjData,
    hand_idx: JointIndexConfig,
    mustard_cfg,
    contact_stats: np.ndarray,
) -> np.ndarray:
    qpos = data.qpos[hand_idx.qpos_ids]
    qvel = data.qvel[hand_idx.dof_ids]
    tcp_xyz = data.site_xpos[hand_idx.tcp_site_ids].reshape(-1)
    mustard_pos = data.qpos[mustard_cfg.qpos_adr : mustard_cfg.qpos_adr + 3]
    mustard_quat = data.qpos[mustard_cfg.qpos_adr + 3 : mustard_cfg.qpos_adr + 7]
    return np.concatenate(
        [qpos, qvel, tcp_xyz, mustard_pos, mustard_quat, contact_stats], axis=0
    ).astype(np.float32)


def _capture_every(control_hz: float, capture_hz: float) -> int:
    if capture_hz <= 0:
        return 1
    ratio = control_hz / capture_hz
    if ratio <= 1.0:
        return 1
    return max(1, int(round(ratio)))


def _save_episode_npz(
    episode_path: Path,
    images: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    phases: list[int],
    contacts: list[np.ndarray],
    success: bool,
    instruction: str,
    side: str,
    control_hz: float,
    capture_hz: float,
    object_qpos: np.ndarray,
    criteria: dict,
    action_semantics: str,
    action_joint16_cmd: np.ndarray | None = None,
    action_tcp12_cmd_world_abs: np.ndarray | None = None,
    action_tcp12_cmd_world_next_delta: np.ndarray | None = None,
    action_tcp12_cmd_palm_local_next_delta: np.ndarray | None = None,
) -> None:
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": np.asarray(images, dtype=np.uint8),
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "phase": np.asarray(phases, dtype=np.int32),
        "contact": np.asarray(contacts, dtype=np.float32),
        "success": np.asarray(success, dtype=np.bool_),
        "language_instruction": np.asarray(instruction),
        "side": np.asarray(side),
        "control_hz": np.asarray(control_hz, dtype=np.float32),
        "capture_hz": np.asarray(capture_hz, dtype=np.float32),
        "object_qpos": np.asarray(object_qpos, dtype=np.float32),
        "criteria_json": np.asarray(json.dumps(criteria), dtype=object),
        "action_semantics": np.asarray(action_semantics, dtype=object),
        "saved_at": np.asarray(datetime.now().isoformat()),
    }
    if action_joint16_cmd is not None:
        payload["action_joint16_cmd"] = np.asarray(action_joint16_cmd, dtype=np.float32)
    if action_tcp12_cmd_world_abs is not None:
        payload["action_tcp12_cmd_world_abs"] = np.asarray(
            action_tcp12_cmd_world_abs, dtype=np.float32
        )
    if action_tcp12_cmd_world_next_delta is not None:
        payload["action_tcp12_cmd_world_next_delta"] = np.asarray(
            action_tcp12_cmd_world_next_delta, dtype=np.float32
        )
    if action_tcp12_cmd_palm_local_next_delta is not None:
        payload["action_tcp12_cmd_palm_local_next_delta"] = np.asarray(
            action_tcp12_cmd_palm_local_next_delta, dtype=np.float32
        )
    np.savez_compressed(episode_path, **payload)


def _fk_tcp_world_from_hand_q(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fk_data: mujoco.MjData,
    hand_idx: JointIndexConfig,
    hand_q_cmd: np.ndarray,
) -> np.ndarray:
    fk_data.qpos[:] = data.qpos
    if fk_data.mocap_pos.size:
        fk_data.mocap_pos[:] = data.mocap_pos
    if fk_data.mocap_quat.size:
        fk_data.mocap_quat[:] = data.mocap_quat
    fk_data.qpos[hand_idx.qpos_ids] = hand_q_cmd
    mujoco.mj_forward(model, fk_data)
    return fk_data.site_xpos[hand_idx.tcp_site_ids].reshape(4, 3).astype(np.float32).copy()


def _tcp_world_abs_to_next_delta(tcp_cmd_world_abs: np.ndarray) -> np.ndarray:
    # tcp_cmd_world_abs: [T, 4, 3]
    if tcp_cmd_world_abs.size == 0:
        return np.zeros((0, 4, 3), dtype=np.float32)
    if tcp_cmd_world_abs.ndim != 3 or tcp_cmd_world_abs.shape[1:] != (4, 3):
        raise ValueError(
            f"Expected tcp_cmd_world_abs shape [T,4,3], got {tcp_cmd_world_abs.shape}"
        )
    if tcp_cmd_world_abs.shape[0] <= 1:
        return np.zeros_like(tcp_cmd_world_abs, dtype=np.float32)
    out = np.zeros_like(tcp_cmd_world_abs, dtype=np.float32)
    out[:-1] = tcp_cmd_world_abs[1:] - tcp_cmd_world_abs[:-1]
    out[-1] = out[-2]
    return out


def _world_delta_to_palm_local_delta(
    tcp_delta_world: np.ndarray,
    palm_rot_body_to_world: np.ndarray,
) -> np.ndarray:
    # row-vector conversion local = world @ R, for each timestep.
    # tcp_delta_world: [T, 4, 3], palm_rot_body_to_world: [T, 3, 3]
    if tcp_delta_world.shape[0] == 0:
        return np.zeros((0, 4, 3), dtype=np.float32)
    if tcp_delta_world.ndim != 3 or tcp_delta_world.shape[1:] != (4, 3):
        raise ValueError(
            f"Expected tcp_delta_world shape [T,4,3], got {tcp_delta_world.shape}"
        )
    if palm_rot_body_to_world.ndim != 3 or palm_rot_body_to_world.shape[1:] != (3, 3):
        raise ValueError(
            "Expected palm_rot_body_to_world shape [T,3,3], got "
            f"{palm_rot_body_to_world.shape}"
        )
    if palm_rot_body_to_world.shape[0] != tcp_delta_world.shape[0]:
        raise ValueError(
            "palm_rot_body_to_world length mismatch: "
            f"{palm_rot_body_to_world.shape[0]} vs {tcp_delta_world.shape[0]}"
        )
    return np.einsum("tfj,tjk->tfk", tcp_delta_world, palm_rot_body_to_world).astype(np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply for [w, x, y, z] convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _apply_spawn_randomization(
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    pos = base_pos.astype(np.float64).copy()
    quat = base_quat.astype(np.float64).copy()

    if args.spawn_jitter_xy > 0.0:
        pos[0] += float(rng.uniform(-args.spawn_jitter_xy, args.spawn_jitter_xy))
        pos[1] += float(rng.uniform(-args.spawn_jitter_xy, args.spawn_jitter_xy))
    if args.spawn_jitter_z > 0.0:
        pos[2] += float(rng.uniform(-args.spawn_jitter_z, args.spawn_jitter_z))
    if args.spawn_yaw_jitter_deg > 0.0:
        yaw = np.deg2rad(
            float(rng.uniform(-args.spawn_yaw_jitter_deg, args.spawn_yaw_jitter_deg))
        )
        qz = np.asarray([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)
        quat = _quat_mul(qz, quat)
        quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return pos.astype(np.float32), quat.astype(np.float32)


def _sample_pose_with_noise(
    base_pose: np.ndarray,
    std: float,
    rng: np.random.Generator,
    q_min: np.ndarray,
    q_max: np.ndarray,
) -> np.ndarray:
    pose = base_pose.astype(np.float64).copy()
    if std > 0.0:
        pose += rng.normal(loc=0.0, scale=std, size=pose.shape)
    return np.clip(pose, q_min, q_max).astype(np.float32)


def _sample_steps(base: int, jitter: int, rng: np.random.Generator) -> int:
    if jitter <= 0:
        return int(base)
    return max(1, int(base + rng.integers(-jitter, jitter + 1)))


def run_collection(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)

    if args.out_dir == DEFAULT_TCP12_OUT_DIR and args.action_interface == "joint16":
        output_root = Path(DEFAULT_FULL_JOINT_OUT_DIR)
    else:
        output_root = Path(args.out_dir)
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    capture_every = _capture_every(args.control_hz, args.capture_hz)
    effective_capture_hz = args.control_hz / capture_every

    mjcf = allegro_hand_mjcf.load(side=args.side, add_mustard=True)
    model = mjcf.compile()
    data = mujoco.MjData(model)
    fk_data = mujoco.MjData(model)

    hand_cfg = build_hand_config(model, args.side)
    hand_idx = _build_joint_indices(model, args.side)
    mustard_cfg = build_mustard_config(model)
    contact_cfg = build_contact_config(model, args.side, mustard_cfg.body_id)

    open_pose = np.clip(np.zeros(16, dtype=np.float32), hand_cfg.q_min, hand_cfg.q_max)
    preshape_pose = np.clip(POWER_GRASP_PRESHAPE_QPOS, hand_cfg.q_min, hand_cfg.q_max).astype(
        np.float32
    )
    close_pose = np.clip(POWER_GRASP_CLOSE_QPOS, hand_cfg.q_min, hand_cfg.q_max).astype(np.float32)

    criteria = {
        "min_contacts": int(args.min_contacts),
        "min_contact_fingers": int(args.min_contact_fingers),
        "require_thumb_contact": bool(args.require_thumb_contact),
        "min_force": float(args.min_force),
        "max_force": float(args.max_force),
        "stable_steps": int(args.stable_steps),
    }

    renderer = mujoco.Renderer(model, width=args.image_width, height=args.image_height)

    viewer_ctx = None
    viewer_delay = (
        args.viewer_step_delay
        if args.viewer_step_delay is not None
        else (0.0 if args.viewer_step <= 0 else 1.0 / args.viewer_step)
    )
    if args.viewer:
        viewer_ctx = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True)
        viewer_ctx.__enter__()
        set_default_hand_camera(viewer_ctx.cam)

    force_buf = np.zeros(6, dtype=float)

    saved_success = 0
    attempts = 0

    try:
        print(
            f"[collect] target={args.target_episodes} max_attempts={args.max_attempts} "
            f"control_hz={args.control_hz:.1f} capture_hz={effective_capture_hz:.1f} "
            f"(capture_every={capture_every}) action_interface={args.action_interface} "
            f"out_dir={output_root}"
        )
        if (
            args.spawn_jitter_xy > 0.0
            or args.spawn_jitter_z > 0.0
            or args.spawn_yaw_jitter_deg > 0.0
            or args.preshape_noise_std > 0.0
            or args.close_noise_std > 0.0
            or args.open_steps_jitter > 0
            or args.preshape_steps_jitter > 0
            or args.close_steps_jitter > 0
            or args.hold_steps_jitter > 0
        ):
            print(
                "[collect] diversity="
                f"spawn_xy±{args.spawn_jitter_xy:.4f}m "
                f"spawn_z±{args.spawn_jitter_z:.4f}m "
                f"spawn_yaw±{args.spawn_yaw_jitter_deg:.1f}deg "
                f"preshape_noise={args.preshape_noise_std:.4f}rad "
                f"close_noise={args.close_noise_std:.4f}rad "
                f"step_jitter(open,preshape,close,hold)="
                f"{args.open_steps_jitter},{args.preshape_steps_jitter},"
                f"{args.close_steps_jitter},{args.hold_steps_jitter}"
            )
        if args.action_interface == "tcp12":
            print(
                "[collect] tcp12 semantics: command next-delta "
                f"(frame={args.tcp12_frame}) from FK(q_cmd_t+1)-FK(q_cmd_t)."
            )
        print(
            "[contact] fingertip_geoms="
            f"{len(contact_cfg.geom_to_finger)} target_geoms={len(contact_cfg.target_geom_ids)}"
        )

        while saved_success < args.target_episodes and attempts < args.max_attempts:
            attempts += 1

            reset_to_initial(model, data)
            spawn_pos_base, spawn_quat_base = compute_mustard_spawn_pose(model, data, args.side)
            spawn_pos, spawn_quat = _apply_spawn_randomization(
                spawn_pos_base, spawn_quat_base, rng, args
            )
            set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
            mujoco.mj_forward(model, data)
            object_qpos = np.concatenate([spawn_pos, spawn_quat], axis=0).astype(np.float32)

            preshape_pose_ep = _sample_pose_with_noise(
                preshape_pose, args.preshape_noise_std, rng, hand_cfg.q_min, hand_cfg.q_max
            )
            close_pose_ep = _sample_pose_with_noise(
                close_pose, args.close_noise_std, rng, hand_cfg.q_min, hand_cfg.q_max
            )
            open_steps_ep = _sample_steps(args.open_steps, args.open_steps_jitter, rng)
            preshape_steps_ep = _sample_steps(args.preshape_steps, args.preshape_steps_jitter, rng)
            close_steps_ep = _sample_steps(args.close_steps, args.close_steps_jitter, rng)
            hold_steps_ep = _sample_steps(args.hold_steps, args.hold_steps_jitter, rng)

            stable_hits = 0
            first_success_step = -1
            best_contacts = 0
            best_force = 0.0
            best_fingers: set[str] = set()
            step_counter = 0

            images: list[np.ndarray] = []
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            actions_joint16_cmd: list[np.ndarray] = []
            actions_tcp12_cmd_world_abs: list[np.ndarray] = []
            actions_palm_rot_body_to_world: list[np.ndarray] = []
            phases: list[int] = []
            contacts: list[np.ndarray] = []

            def _step(q_cmd: np.ndarray, phase_id: int) -> None:
                nonlocal stable_hits
                nonlocal first_success_step
                nonlocal best_contacts
                nonlocal best_force
                nonlocal best_fingers
                nonlocal step_counter

                _, n_contacts_pre, total_force_pre, touched_fingers_pre = detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )

                if step_counter % capture_every == 0:
                    thumb_contact = 1.0 if "th" in touched_fingers_pre else 0.0
                    contact_stats = np.asarray(
                        [
                            float(n_contacts_pre),
                            float(total_force_pre),
                            float(len(touched_fingers_pre)),
                            thumb_contact,
                        ],
                        dtype=np.float32,
                    )
                    images.append(_capture_frame(renderer, data))
                    states.append(_make_state_vector(data, hand_idx, mustard_cfg, contact_stats))
                    q_cmd_vec = q_cmd.astype(np.float32).copy()
                    tcp_cmd_world = _fk_tcp_world_from_hand_q(
                        model=model,
                        data=data,
                        fk_data=fk_data,
                        hand_idx=hand_idx,
                        hand_q_cmd=q_cmd_vec,
                    )

                    actions_joint16_cmd.append(q_cmd_vec.copy())
                    actions_tcp12_cmd_world_abs.append(tcp_cmd_world.astype(np.float32).copy())
                    actions_palm_rot_body_to_world.append(
                        data.xmat[hand_idx.palm_body_id].reshape(3, 3).astype(np.float32).copy()
                    )
                    if args.action_interface == "tcp12":
                        # Temporarily store absolute command TCPs and convert to next-delta after episode.
                        action_vec = tcp_cmd_world.reshape(-1).astype(np.float32)
                    else:
                        action_vec = q_cmd_vec
                    actions.append(action_vec.copy())
                    phases.append(int(phase_id))
                    contacts.append(contact_stats.copy())

                data.ctrl[:16] = q_cmd
                set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_step(model, data)
                set_mustard_pose(data, mustard_cfg, spawn_pos, spawn_quat)
                mujoco.mj_forward(model, data)

                _, n_contacts, total_force, touched_fingers = detect_contact_with_target(
                    model, data, contact_cfg, force_buf
                )
                best_contacts = max(best_contacts, int(n_contacts))
                best_force = max(best_force, float(total_force))
                if len(touched_fingers) >= len(best_fingers):
                    best_fingers = set(touched_fingers)

                thumb_ok = (not args.require_thumb_contact) or ("th" in touched_fingers)
                meets = (
                    n_contacts >= args.min_contacts
                    and len(touched_fingers) >= args.min_contact_fingers
                    and thumb_ok
                    and total_force >= args.min_force
                    and total_force <= args.max_force
                )
                stable_hits = stable_hits + 1 if meets else 0
                if stable_hits >= args.stable_steps and first_success_step < 0:
                    first_success_step = step_counter

                if viewer_ctx is not None:
                    viewer_ctx.sync()
                    if viewer_delay > 0.0:
                        time.sleep(viewer_delay)

                step_counter += 1

            for _ in range(open_steps_ep):
                _step(open_pose, PHASE_OPEN)
            for i in range(preshape_steps_ep):
                alpha = (i + 1) / max(preshape_steps_ep, 1)
                cmd = (1.0 - alpha) * open_pose + alpha * preshape_pose_ep
                _step(cmd.astype(np.float32), PHASE_PRESHAPE)
            for i in range(close_steps_ep):
                alpha = (i + 1) / max(close_steps_ep, 1)
                cmd = (1.0 - alpha) * preshape_pose_ep + alpha * close_pose_ep
                _step(cmd.astype(np.float32), PHASE_CLOSE)
            for _ in range(hold_steps_ep):
                _step(close_pose_ep, PHASE_HOLD)

            success = first_success_step >= 0
            if success:
                tcp_cmd_world_abs = np.asarray(actions_tcp12_cmd_world_abs, dtype=np.float32)
                tcp_world_next_delta = _tcp_world_abs_to_next_delta(tcp_cmd_world_abs)
                palm_rot = np.asarray(actions_palm_rot_body_to_world, dtype=np.float32)
                tcp_palm_local_next_delta = _world_delta_to_palm_local_delta(
                    tcp_delta_world=tcp_world_next_delta,
                    palm_rot_body_to_world=palm_rot,
                )

                if args.action_interface == "tcp12":
                    if args.tcp12_frame == "palm_local":
                        actions_to_save = tcp_palm_local_next_delta.reshape(-1, 12)
                    else:
                        actions_to_save = tcp_world_next_delta.reshape(-1, 12)
                    action_semantics = (
                        "tcp12_cmd_next_delta_palm_local"
                        if args.tcp12_frame == "palm_local"
                        else "tcp12_cmd_next_delta_world"
                    )
                else:
                    actions_to_save = actions
                    action_semantics = "joint16_absolute_target"
                ep_name = f"episode_{saved_success:05d}.npz"
                ep_path = raw_dir / ep_name
                _save_episode_npz(
                    episode_path=ep_path,
                    images=images,
                    states=states,
                    actions=actions_to_save,
                    phases=phases,
                    contacts=contacts,
                    success=success,
                    instruction=args.instruction,
                    side=args.side,
                    control_hz=float(args.control_hz),
                    capture_hz=float(effective_capture_hz),
                    object_qpos=object_qpos,
                    criteria=criteria,
                    action_semantics=action_semantics,
                    action_joint16_cmd=np.asarray(actions_joint16_cmd, dtype=np.float32),
                    action_tcp12_cmd_world_abs=np.asarray(actions_tcp12_cmd_world_abs, dtype=np.float32),
                    action_tcp12_cmd_world_next_delta=np.asarray(
                        tcp_world_next_delta.reshape(-1, 12), dtype=np.float32
                    ),
                    action_tcp12_cmd_palm_local_next_delta=np.asarray(
                        tcp_palm_local_next_delta.reshape(-1, 12), dtype=np.float32
                    ),
                )
                saved_success += 1
                status = "✅ SAVE"
            else:
                status = "❌ SKIP"

            print(
                f"[attempt {attempts:04d}] {status} success={int(success)} "
                f"saved={saved_success}/{args.target_episodes} "
                f"contacts={best_contacts} fingers={','.join(sorted(best_fingers)) or '-'} "
                f"force={best_force:.2f}"
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
        "control_hz": float(args.control_hz),
        "capture_hz": float(effective_capture_hz),
        "capture_every": int(capture_every),
        "action_interface": args.action_interface,
        "tcp12_frame": args.tcp12_frame if args.action_interface == "tcp12" else None,
        "action_semantics": (
            (
                "tcp12_cmd_next_delta_palm_local"
                if args.tcp12_frame == "palm_local"
                else "tcp12_cmd_next_delta_world"
            )
            if args.action_interface == "tcp12"
            else "joint16_absolute_target"
        ),
        "image_shape": [args.image_height, args.image_width, 3],
        "state_dim": 16 + 16 + 12 + 7 + 4,
        "action_dim": 12 if args.action_interface == "tcp12" else 16,
        "criteria": criteria,
        "diversity": {
            "spawn_jitter_xy": float(args.spawn_jitter_xy),
            "spawn_jitter_z": float(args.spawn_jitter_z),
            "spawn_yaw_jitter_deg": float(args.spawn_yaw_jitter_deg),
            "preshape_noise_std": float(args.preshape_noise_std),
            "close_noise_std": float(args.close_noise_std),
            "open_steps_jitter": int(args.open_steps_jitter),
            "preshape_steps_jitter": int(args.preshape_steps_jitter),
            "close_steps_jitter": int(args.close_steps_jitter),
            "hold_steps_jitter": int(args.hold_steps_jitter),
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
